#pragma once

#include "terra/fe/wedge/sl/ghost_exchange.hpp"
#include "terra/fe/wedge/sl/point_location.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include <limits>
#include <type_traits>

namespace terra::fe::wedge::sl {

template < typename T >
struct LocatedSample
{
    int subdomain = -1;
    WedgeCell cell{}; // Owned, unghosted cell indices.
    T xi{}, eta{}, zeta{};
};

// Translate the existing interpolators' field accesses into a team-local stencil cache.
template < typename T, typename Scratch >
struct ScratchField
{
    Scratch values;
    int x0, y0, r0, ny, nr, components, component;
    KOKKOS_INLINE_FUNCTION T operator()( int, int x, int y, int r ) const
    {
        return values( ( ( ( x - x0 ) * ny + y - y0 ) * nr + r - r0 ) * components + component );
    }
    KOKKOS_INLINE_FUNCTION T operator()( int s, int x, int y, int r, int d ) const
    {
        auto field = *this;
        field.component += d;
        return field( s, x, y, r );
    }
};

template < typename T, typename Scratch >
struct ScratchRadii
{
    Scratch values;
    int offset, first;
    KOKKOS_INLINE_FUNCTION T operator()( int, int r ) const { return values( offset + r - first ); }
};

// Binning and scratch storage are execution details; both paths use exactly the same field evaluator.
template < typename T >
class TiledInterpolator
{
  public:
    using Vec3 = dense::Vec< T, 3 >;
    static constexpr int tile_cells = 4;
    using Policy = Kokkos::TeamPolicy<>;
    using Team = typename Policy::member_type;
    using Scratch = Kokkos::View< T*, typename Team::scratch_memory_space, Kokkos::MemoryUnmanaged >;

    TiledInterpolator( const grid::shell::DistributedDomain& domain,
                       grid::Grid4DDataScalar< T > temperature,
                       grid::Grid4DDataVec< T, 3 > velocity,
                       grid::Grid4DDataVec< T, 3 > previous_velocity,
                       grid::Grid2DDataScalar< T > radii,
                       Kokkos::View< uint8_t*** > validity )
        : temperature_( temperature ), velocity_( velocity ), previous_velocity_( previous_velocity ),
          radii_( radii ), validity_( validity ),
          lateral_cells_( domain.domain_info().subdomain_num_nodes_per_side_laterally() - 1 ),
          radial_cells_( domain.domain_info().subdomain_num_nodes_radially() - 1 ),
          lateral_tiles_( ( lateral_cells_ + tile_cells - 1 ) / tile_cells ),
          radial_tiles_( ( radial_cells_ + tile_cells - 1 ) / tile_cells )
    {
        const std::size_t bins = domain.subdomains().size() * std::size_t( lateral_tiles_ ) * lateral_tiles_ * radial_tiles_;
        if ( bins > std::size_t( std::numeric_limits< int >::max() ) ) MPI_Abort( domain.comm(), 1 );
        bins_ = int( bins );
        counts_ = decltype( counts_ )( "mmoc_tile_counts", bins );
        offsets_ = decltype( offsets_ )( "mmoc_tile_offsets", bins + 1 );
        cursors_ = decltype( cursors_ )( "mmoc_tile_cursors", bins );
        active_ = decltype( active_ )( "mmoc_active_tiles", bins );
        stencils_ = decltype( stencils_ )( "mmoc_stencil_bounds", domain.subdomains().size() );
        auto host = Kokkos::create_mirror_view( stencils_ );
        const int lateral_domains = domain.domain_info().num_subdomains_per_diamond_side();
        const int radial_domains = domain.domain_info().num_subdomains_in_radial_direction();
        for ( const auto& [id, data] : domain.subdomains() )
        {
            // Ghosts extend a stencil inside the same diamond. Across a diamond seam the index chart kinks.
            host( std::get< 0 >( data ) ) = {
                { id.subdomain_x() > 0 ? 0 : ghost_width,
                  lateral_cells_ + ghost_width + ( id.subdomain_x() + 1 < lateral_domains ? ghost_width : 0 ) },
                { id.subdomain_y() > 0 ? 0 : ghost_width,
                  lateral_cells_ + ghost_width + ( id.subdomain_y() + 1 < lateral_domains ? ghost_width : 0 ) },
                { id.subdomain_r() > 0 ? 0 : ghost_width,
                  radial_cells_ + ghost_width + ( id.subdomain_r() + 1 < radial_domains ? ghost_width : 0 ) } };
        }
        Kokkos::deep_copy( stencils_, host );
    }

    std::size_t buffer_bytes() const
    {
        return ( counts_.span() + offsets_.span() + cursors_.span() + active_.span() + indices_.span() ) * sizeof( int ) +
            stencils_.span() * sizeof( StencilBounds );
    }

    template < bool Velocity >
    void evaluate( const Kokkos::View< LocatedSample< T >* >& locations, int count,
                   const Kokkos::View< std::conditional_t< Velocity, Vec3, T >* >& output,
                   T tau = T( 0 ), bool shared = true )
    {
        const auto temperature = temperature_;
        const auto velocity = velocity_;
        const auto previous_velocity = previous_velocity_;
        const auto radii = radii_;
        const auto validity = validity_;
        const auto stencils = stencils_;
        if ( !shared )
        {
            Kokkos::parallel_for( "mmoc_interpolate_global", count,
                [locations, velocity, previous_velocity, output, tau, temperature, radii, stencils, validity]
                KOKKOS_FUNCTION( int i ) {
                const auto q = locations( i );
                if ( q.subdomain < 0 ) return;
                auto cell = q.cell; cell.x += ghost_width; cell.y += ghost_width; cell.r += ghost_width;
                if constexpr ( Velocity )
                {
                    const auto a = evaluate_q1_vec< T, 3 >( velocity, q.subdomain, cell, q.xi, q.eta, q.zeta );
                    const auto b = evaluate_q1_vec< T, 3 >( previous_velocity, q.subdomain, cell, q.xi, q.eta, q.zeta );
                    output( i ) = a * ( T( 1 ) - tau ) + b * tau;
                }
                else output( i ) = evaluate_cubic_scalar( temperature, q.subdomain, cell, q.xi, q.eta, q.zeta,
                                                         radii, stencils( q.subdomain ), validity, true );
            } );
            Kokkos::fence();
            return;
        }
        if ( indices_.extent( 0 ) < std::size_t( count ) ) indices_ = decltype( indices_ )( "mmoc_tile_indices", count );
        const auto counts = counts_, offsets = offsets_, cursors = cursors_, indices = indices_, active = active_;
        const int nt = lateral_tiles_, rt = radial_tiles_, nc = lateral_cells_, rc = radial_cells_;
        Kokkos::deep_copy( counts, 0 );
        Kokkos::deep_copy( cursors, 0 );
        Kokkos::parallel_for( "mmoc_tile_histogram", count, KOKKOS_LAMBDA( int i ) {
            const auto q = locations( i );
            if ( q.subdomain < 0 ) return;
            const int bin = ( ( q.subdomain * nt + q.cell.x / tile_cells ) * nt + q.cell.y / tile_cells ) * rt + q.cell.r / tile_cells;
            Kokkos::atomic_add( &counts( bin ), 1 );
        } );
        const int num_bins = bins_;
        Kokkos::parallel_scan( "mmoc_tile_offsets", bins_, KOKKOS_LAMBDA( int i, int& sum, bool final ) {
            if ( final ) offsets( i ) = sum;
            sum += counts( i );
            if ( final && i + 1 == num_bins ) offsets( i + 1 ) = sum;
        } );
        Kokkos::parallel_for( "mmoc_tile_scatter", count, KOKKOS_LAMBDA( int i ) {
            const auto q = locations( i );
            if ( q.subdomain < 0 ) return;
            const int bin = ( ( q.subdomain * nt + q.cell.x / tile_cells ) * nt + q.cell.y / tile_cells ) * rt + q.cell.r / tile_cells;
            indices( offsets( bin ) + Kokkos::atomic_fetch_add( &cursors( bin ), 1 ) ) = i;
        } );
        int active_count = 0;
        Kokkos::parallel_scan( "mmoc_active_tiles", bins_, KOKKOS_LAMBDA( int i, int& sum, bool final ) {
            if ( counts( i ) > 0 ) { if ( final ) active( sum ) = i; ++sum; }
        }, active_count );
        // A tile needs at most (4+3)^3 cubic nodes; velocity uses at most (4+1)^3 nodes for two vectors.
        constexpr int cache_values = Velocity ? 5 * 5 * 5 * 6 : 7 * 7 * 7;
        constexpr int scratch_values = cache_values + ( Velocity ? 0 : 7 );
        Policy policy( active_count, Kokkos::AUTO );
#ifdef KOKKOS_ENABLE_CUDA
        if constexpr ( std::is_same_v< Kokkos::DefaultExecutionSpace, Kokkos::Cuda > ) policy = Policy( active_count, 128 );
#endif
        policy.set_scratch_size( 0, Kokkos::PerTeam( Scratch::shmem_size( scratch_values ) ) );
        Kokkos::parallel_for( Velocity ? "mmoc_velocity_shared" : "mmoc_temperature_shared", policy,
            [active, nt, rt, nc, rc, stencils, cache_values, scratch_values, velocity, previous_velocity,
             temperature, radii, offsets, indices, locations, output, tau, validity]
            KOKKOS_FUNCTION( const Team& team ) {
                const int bin = active( team.league_rank() );
                const int s = bin / ( nt * nt * rt );
                const int tx = ( bin / ( nt * rt ) ) % nt, ty = ( bin / rt ) % nt, tr = bin % rt;
                const int cx0 = tx * tile_cells + ghost_width, cy0 = ty * tile_cells + ghost_width,
                          cr0 = tr * tile_cells + ghost_width;
                const int cx1 = Kokkos::min( ( tx + 1 ) * tile_cells, nc ) - 1 + ghost_width;
                const int cy1 = Kokkos::min( ( ty + 1 ) * tile_cells, nc ) - 1 + ghost_width;
                const int cr1 = Kokkos::min( ( tr + 1 ) * tile_cells, rc ) - 1 + ghost_width;
                int x0 = cx0, y0 = cy0, r0 = cr0, x1 = cx1 + 1, y1 = cy1 + 1, r1 = cr1 + 1;
                if constexpr ( !Velocity )
                {
                    const auto bounds = stencils( s );
                    int end;
                    cubic_stencil_window( cx0, bounds.x, x0 );
                    const int nx = cubic_stencil_window( cx1, bounds.x, end ); x1 = end + nx - 1;
                    cubic_stencil_window( cy0, bounds.y, y0 );
                    const int ny = cubic_stencil_window( cy1, bounds.y, end ); y1 = end + ny - 1;
                    cubic_stencil_window( cr0, bounds.r, r0 );
                    const int nr = cubic_stencil_window( cr1, bounds.r, end ); r1 = end + nr - 1;
                }
                const int nx = x1 - x0 + 1, ny = y1 - y0 + 1, nr = r1 - r0 + 1;
                constexpr int components = Velocity ? 6 : 1;
                KOKKOS_ASSERT( nx * ny * nr * components <= cache_values );
                Scratch cache( team.team_scratch( 0 ), scratch_values );
                Kokkos::parallel_for( Kokkos::TeamThreadRange( team, nx * ny * nr * components ), [&]( int i ) {
                    const int c = i % components, node = i / components;
                    const int x = x0 + node / ( ny * nr ), y = y0 + ( node / nr ) % ny, r = r0 + node % nr;
                    if constexpr ( Velocity ) cache( i ) = c < 3 ? velocity( s, x, y, r, c ) : previous_velocity( s, x, y, r, c - 3 );
                    else cache( i ) = temperature( s, x, y, r );
                } );
                if constexpr ( !Velocity )
                    Kokkos::parallel_for( Kokkos::TeamThreadRange( team, nr ), [&]( int i ) {
                        cache( cache_values + i ) = radii( s, r0 + i );
                    } );
                team.team_barrier();
                Kokkos::parallel_for( Kokkos::TeamThreadRange( team, offsets( bin ), offsets( bin + 1 ) ), [&]( int j ) {
                    const int i = indices( j ); const auto q = locations( i );
                    auto cell = q.cell; cell.x += ghost_width; cell.y += ghost_width; cell.r += ghost_width;
                    const ScratchField< T, Scratch > field{ cache, x0, y0, r0, ny, nr, components, 0 };
                    if constexpr ( Velocity )
                    {
                        auto previous = field; previous.component = 3;
                        const auto a = evaluate_q1_vec< T, 3 >( field, s, cell, q.xi, q.eta, q.zeta );
                        const auto b = evaluate_q1_vec< T, 3 >( previous, s, cell, q.xi, q.eta, q.zeta );
                        output( i ) = a * ( T( 1 ) - tau ) + b * tau;
                    }
                    else
                    {
                        const ScratchRadii< T, Scratch > cached_radii{ cache, cache_values, r0 };
                        output( i ) = evaluate_cubic_scalar( field, s, cell, q.xi, q.eta, q.zeta,
                                                            cached_radii, stencils( s ), validity, true );
                    }
                } );
            } );
        Kokkos::fence();
    }

  private:
    grid::Grid4DDataScalar< T > temperature_;
    grid::Grid4DDataVec< T, 3 > velocity_, previous_velocity_;
    grid::Grid2DDataScalar< T > radii_;
    Kokkos::View< uint8_t*** > validity_;
    int lateral_cells_, radial_cells_, lateral_tiles_, radial_tiles_, bins_;
    Kokkos::View< int* > counts_, offsets_, cursors_, indices_, active_;
    Kokkos::View< StencilBounds* > stencils_;
};
} // namespace terra::fe::wedge::sl
