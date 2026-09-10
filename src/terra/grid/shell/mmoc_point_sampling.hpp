#pragma once

#include "terra/communication/shell/point_queries.hpp"
#include "terra/fe/wedge/sl/tiled_interpolation.hpp"
#include "terra/grid/bit_masks.hpp"
#include "terra/grid/shell/lateral_cell_lookup.hpp"
#include "terra/grid/shell/spherical_shell.hpp"

#include <bit>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <cstdint>
#include <functional>
#include <map>
#include <type_traits>
#include <utility>

namespace terra::grid::shell {

/// Resolve batches of physical sampling positions on their owning subdomains.
/// Scalar and vector evaluation use the same routing and communication buffers. All ranks in domain.comm()
/// participate; positions and results use flattened local node indices. Only owning nodes generate queries.
/// The caller supplies a batched evaluator and synchronizes shared output nodes after the final sample.
template < std::floating_point T >
class MMOCPointSampler
{
  public:
    // CUDA requires public record types in Kokkos views captured by device kernels.
    struct Query
    {
        dense::Vec< T, 3 > point{}; // Physical position from the planet center; norm = radius.
        int origin_node = -1; // Flattened output-node index on the requesting rank.
        // Destination identifiers are global; the receiving rank resolves its local index.
        int diamond_id = -1;
        int subdomain_x = -1;
        int subdomain_y = -1;
        int subdomain_r = -1;
        int local_subdomain = -1;
    };

    // Return the sampled field value to the original output node.
    template < typename Value >
    struct Reply
    {
        int origin_node = -1;
        Value value{};
    };

    // Only locally owned subdomains are stored; metadata scales with this rank's partition.
    struct LocalSubdomainLookup
    {
        Grid1DDataScalar< int64_t > keys;
        Grid1DDataScalar< int > indices;
        int lateral = 0;
        int radial = 0;

        KOKKOS_INLINE_FUNCTION int64_t key( const Query& query ) const
        {
            return ( ( int64_t( query.diamond_id ) * lateral + query.subdomain_x ) * lateral +
                     query.subdomain_y ) * radial + query.subdomain_r;
        }

        KOKKOS_INLINE_FUNCTION int operator()( const Query& query ) const
        {
            // Sorted device metadata replaces a host map lookup for every departure.
            const auto target = key( query );
            int lo = 0, hi = keys.extent( 0 );
            while ( lo < hi )
            {
                const int mid = lo + ( hi - lo ) / 2;
                if ( keys( mid ) < target ) lo = mid + 1;
                else hi = mid;
            }
            return lo < int( keys.extent( 0 ) ) && keys( lo ) == target ? indices( lo ) : -1;
        }

        // Radial segments of a lateral patch share the same unit-sphere geometry.
        KOKKOS_INLINE_FUNCTION int lateral_subdomain( const Query& query ) const
        {
            const auto first = key( query ) - query.subdomain_r;
            int lo = 0, hi = keys.extent( 0 );
            while ( lo < hi )
            {
                const int mid = lo + ( hi - lo ) / 2;
                if ( keys( mid ) < first ) lo = mid + 1;
                else hi = mid;
            }
            return lo < int( keys.extent( 0 ) ) && keys( lo ) < first + radial ? indices( lo ) : -1;
        }
    };

    // The direct locator sees a complete diamond. Stored nodes are read locally; missing nodes
    // are reconstructed from root corners, without a geometry exchange or global fine-grid allocation.
    struct RoutingCoordinates
    {
        Grid3DDataVec< T, 3 > coords;
        LocalSubdomainLookup lookup;
        detail::ReconstructedDiamondCoordinates< T > reconstructed;
        int diamond;
        int local_cells;

        KOKKOS_INLINE_FUNCTION dense::Vec< T, 3 > node_position( int, int x, int y ) const
        {
            Query query;
            query.diamond_id = diamond;
            query.subdomain_x = Kokkos::min( x / local_cells, lookup.lateral - 1 );
            query.subdomain_y = Kokkos::min( y / local_cells, lookup.lateral - 1 );
            const int s = lookup.lateral_subdomain( query );
            if ( s < 0 )
                return reconstructed.node_position( 0, x, y );
            dense::Vec< T, 3 > node;
            for ( int d = 0; d < 3; ++d )
                node( d ) = coords( s, x - query.subdomain_x * local_cells,
                                      y - query.subdomain_y * local_cells, d );
            return node;
        }
    };

  private:
    const DistributedDomain& domain_;
    SubdomainToRankDistributionFunction subdomain_to_rank_;
    Grid3DDataVec< T, 3 > coords_;
    Grid2DDataScalar< T > radii_;
    Grid4DDataScalar< NodeOwnershipFlag > ownership_;
    // Routing geometry: ten diamond roots and radial partition bounds.
    Grid3DDataVec< T, 3 > diamond_corners_;
    Grid1DDataScalar< T > radial_bounds_;
    LocalSubdomainLookup local_lookup_;
    // Remote owners are evaluated once per destination, without replicating a global table.
    std::map< int64_t, int > destination_ranks_;
    // One outgoing queue per rank, shared by all local subdomains; incoming storage grows on demand.
    Kokkos::View< Query* > outgoing_;
    Kokkos::View< Query* > incoming_;
    // Reused compaction buffer: only queries leaving this MPI rank are copied to the host.
    Kokkos::View< Query* > remote_;
    Kokkos::View< fe::wedge::sl::LocatedSample< T >* > locations_;
    Kokkos::View< T* > scalar_values_;
    Kokkos::View< dense::Vec< T, 3 >* > vector_values_;
    Kokkos::View< Reply< T >* > scalar_replies_;
    Kokkos::View< Reply< dense::Vec< T, 3 > >* > vector_replies_;
    Kokkos::View< Query*, Kokkos::HostSpace > outgoing_host_, incoming_host_;
    std::map< int, std::vector< Query > > batches_;
    Kokkos::View< int > outgoing_count_{ "outgoing_point_count" };
    Kokkos::View< int > remote_count_{ "remote_point_count" };
    // Compact only departures whose physical radial owner is ambiguous at partition resolution.
    Kokkos::View< int* > radial_boundary_queries_;
    Kokkos::View< int > radial_boundary_count_{ "radial_boundary_point_count" };
    // Counts invalid positions or failed containment, not normal boundary crossings.
    Kokkos::View< int > errors_{ "point_lookup_errors" };
    int subdivision_level_;
    int node_count_;
    size_t last_remote_queries_ = 0;

    void require( bool condition, const char* message ) const
    {
        communication::shell::detail::point_query_require( condition, domain_.comm(), message );
    }

    // Read kernel failures on the host and stop MPI before invalid results are used.
    void check_kernel_errors() const
    {
        int errors = 0;
        Kokkos::deep_copy( errors, errors_ );
        require( errors == 0, "point lies outside the shell or could not be located" );
    }

  public:
    MMOCPointSampler(
        const DistributedDomain& domain,
        const Grid3DDataVec< T, 3 >& coords,
        const Grid2DDataScalar< T >& radii,
        const Grid4DDataScalar< NodeOwnershipFlag >& ownership,
        SubdomainToRankDistributionFunction subdomain_to_rank )
    : domain_( domain )
    , subdomain_to_rank_( std::move( subdomain_to_rank ) )
    , coords_( coords )
    , radii_( radii )
    , ownership_( ownership )
    , diamond_corners_( "point_lookup_diamond_roots", 10, 2, 2 )
    , radial_bounds_( "point_lookup_radial_subdomain_bounds", domain.domain_info().num_subdomains_in_radial_direction() + 1 )
    {
        const auto& info = domain.domain_info();
        const auto n = static_cast< unsigned >( info.subdomain_num_nodes_per_side_laterally() - 1 );
        const auto subdivisions = static_cast< unsigned >( info.num_subdomains_per_diamond_side() );
        require( std::has_single_bit( n ) && std::has_single_bit( subdivisions ), "non-dyadic lateral grid" );
        subdivision_level_ = std::countr_zero( subdivisions );
        node_count_ = communication::shell::detail::point_query_count( ownership.size(), domain.comm() );
        outgoing_ = Kokkos::View< Query* >( "outgoing_point_queries", node_count_ );
        locations_ = decltype( locations_ )( "mmoc_locations", node_count_ );

        auto corners_host = Kokkos::create_mirror_view( diamond_corners_ );
        for ( int diamond = 0; diamond < 10; ++diamond )
            unit_sphere_single_shell_subdomain_coords< T >( corners_host, diamond, diamond, 2, 0, 1, 0, 1 );
        Kokkos::deep_copy( diamond_corners_, corners_host );
        auto bounds_host = Kokkos::create_mirror_view( radial_bounds_ );
        const int radial_stride = info.subdomain_num_nodes_radially() - 1;
        for ( size_t i = 0; i < radial_bounds_.extent( 0 ); ++i )
            bounds_host( i ) = info.radii()[i * radial_stride];
        Kokkos::deep_copy( radial_bounds_, bounds_host );

        // The partition is fixed for this sampler, so build its device lookup only once.
        local_lookup_.lateral = subdivisions;
        local_lookup_.radial = info.num_subdomains_in_radial_direction();
        std::vector< std::pair< int64_t, int > > local_indices;
        for ( const auto& [id, data] : domain.subdomains() )
        {
            Query query;
            query.diamond_id = id.diamond_id();
            query.subdomain_x = id.subdomain_x();
            query.subdomain_y = id.subdomain_y();
            query.subdomain_r = id.subdomain_r();
            local_indices.emplace_back( local_lookup_.key( query ), std::get< 0 >( data ) );
        }
        std::sort( local_indices.begin(), local_indices.end() );
        local_lookup_.keys = Grid1DDataScalar< int64_t >( "point_lookup_local_keys", local_indices.size() );
        local_lookup_.indices = Grid1DDataScalar< int >( "point_lookup_local_indices", local_indices.size() );
        auto keys_host = Kokkos::create_mirror_view( local_lookup_.keys );
        auto indices_host = Kokkos::create_mirror_view( local_lookup_.indices );
        for ( size_t i = 0; i < local_indices.size(); ++i )
        {
            keys_host( i ) = local_indices[i].first;
            indices_host( i ) = local_indices[i].second;
        }
        Kokkos::deep_copy( local_lookup_.keys, keys_host );
        Kokkos::deep_copy( local_lookup_.indices, indices_host );
    }

    std::size_t last_remote_queries() const { return last_remote_queries_; }
    std::size_t buffer_bytes() const
    {
        return outgoing_.span() * sizeof( Query ) + incoming_.span() * sizeof( Query ) +
            remote_.span() * sizeof( Query ) + locations_.span() * sizeof( fe::wedge::sl::LocatedSample< T > ) +
            scalar_values_.span() * sizeof( T ) + vector_values_.span() * sizeof( dense::Vec< T, 3 > ) +
            radial_boundary_queries_.span() * sizeof( int ) + scalar_replies_.span() * sizeof( Reply< T > ) +
            vector_replies_.span() * sizeof( Reply< dense::Vec< T, 3 > > );
    }

    // Locate only in owned cells. Ghost values are exclusively an interpolation stencil resource.
    KOKKOS_INLINE_FUNCTION static fe::wedge::sl::LocatedSample< T > locate_owned(
        const Grid3DDataVec< T, 3 >& coords, const Grid2DDataScalar< T >& radii,
        int subdomain, const dense::Vec< T, 3 >& point, T inner, T outer )
    {
        namespace sl = fe::wedge::sl;
        sl::LocatedSample< T > result;
        if ( !( point.norm() > T( 0 ) ) || !Kokkos::isfinite( point.norm() ) ) return result;
        const sl::IndexBounds bounds{ int( coords.extent( 1 ) ), int( coords.extent( 2 ) ), int( radii.extent( 1 ) ) };
        const auto found = sl::locate_point_direct( point, subdomain, coords, radii,
            sl::corner_box_from_bounds( bounds ), bounds, 2, 4, T( 1e-12 ), true, inner, outer,
            detail::ValidLateralNodes{} );
        if ( !found.found || found.zeta < T( -1 ) - T( 1e-11 ) || found.zeta > T( 1 ) + T( 1e-11 ) ) return result;
        result.subdomain = subdomain;
        result.cell = found.cell;
        result.xi = found.xi; result.eta = found.eta;
        result.zeta = Kokkos::clamp( found.zeta, T( -1 ), T( 1 ) );
        return result;
    }

    // Every rank participates, including ranks with no owned nodes or outgoing requests.
    // evaluate batches located samples on the GPU and writes results at their original indices.
    // Keep the host callback type out of the enclosing CUDA kernel template: NVCC cannot
    // instantiate extended lambdas when a parent template argument is a function-local lambda.
    template < typename Value >
    void sample( const Kokkos::View< dense::Vec< T, 3 >* >& positions,
                 const Kokkos::View< Value* >& output,
                 const std::type_identity_t< std::function< void(
                     const Kokkos::View< fe::wedge::sl::LocatedSample< T >* >&, int,
                     const Kokkos::View< Value* >& ) > >& evaluate )
    {
        Kokkos::Profiling::ScopedRegion sampling_region( "mmoc_point_sampling" );
        require( positions.extent( 0 ) == std::size_t( node_count_ ) && output.extent( 0 ) == positions.extent( 0 ),
                 "MMOC position/result shape mismatch" );
        const auto coords = coords_;
        const auto radii = radii_;
        const auto ownership = ownership_;
        const auto outgoing = outgoing_;
        const auto outgoing_count = outgoing_count_;
        const auto errors = errors_;
        const auto locations = locations_;
        const int n = ownership.extent( 1 );
        const int nr = ownership.extent( 3 );
        const auto radial_bounds = radial_bounds_;
        constexpr bool physical_wedge = true;
        Kokkos::deep_copy( outgoing_count, 0 );
        Kokkos::deep_copy( errors, 0 );
        Kokkos::deep_copy( output, Value{} );
        Kokkos::parallel_for( "mmoc_classify_owned", node_count_, KOKKOS_LAMBDA( int node ) {
            locations( node ).subdomain = -1;
            const int s = node / ( n * n * nr );
            const int x = ( node / ( n * nr ) ) % n, y = ( node / nr ) % n, r = node % nr;
            if ( !util::has_flag( ownership( s, x, y, r ), NodeOwnershipFlag::OWNED ) ) return;
            const auto point = positions( node );
            const auto found = locate_owned( coords, radii, s, point, radial_bounds( 0 ),
                                            radial_bounds( radial_bounds.extent( 0 ) - 1 ) );
            if ( found.subdomain >= 0 )
            {
                locations( node ) = found;
                return;
            }
            Query query;
            query.point = point; query.origin_node = node;
            outgoing( Kokkos::atomic_fetch_add( &outgoing_count(), 1 ) ) = query;
        } );
        int count = 0;
        Kokkos::deep_copy( count, outgoing_count );
        const auto diamond_corners = diamond_corners_;
        const auto local_lookup = local_lookup_;
        const int subdivision_level = subdivision_level_;
        // 2. Find complete destination IDs at partition resolution. For a unit-vertex mesh,
        // |point| <= physical fine-wedge radius <= containing coarse-wedge radius.
        // Most departures therefore need no fine geometry to identify their radial owner.
        if ( physical_wedge && radial_boundary_queries_.extent( 0 ) < size_t( count ) )
            radial_boundary_queries_ = Kokkos::View< int* >( "radial_boundary_point_queries", count );
        const auto radial_boundary_queries = radial_boundary_queries_;
        const auto radial_boundary_count = radial_boundary_count_;
        if ( physical_wedge )
            Kokkos::deep_copy( radial_boundary_count, 0 );
        Kokkos::parallel_for(
            "point_queries_resolve_destination_chart", count,
            KOKKOS_LAMBDA( int i ) {
                auto& query = outgoing( i );
                const T radius = query.point.norm();
                const int radial_domains = radial_bounds.extent( 0 ) - 1;
                const T tolerance = T( 64 ) * std::numeric_limits< T >::epsilon() * radial_bounds( radial_domains );
                if ( !Kokkos::isfinite( radius ) || !( radius > T( 0 ) ) ||
                     ( !physical_wedge && ( radius < radial_bounds( 0 ) - tolerance ||
                                            radius > radial_bounds( radial_domains ) + tolerance ) ) )
                {
                    Kokkos::atomic_add( &errors(), 1 );
                    return;
                }
                for ( int diamond = 0; diamond < 10; ++diamond )
                {
                    dense::Vec< T, 3 > corners[4];
                    for ( int c = 0; c < 4; ++c )
                        for ( int d = 0; d < 3; ++d )
                            corners[c]( d ) = diamond_corners( diamond, c % 2, c / 2, d );
                    T coarse_radius = radius;
                    const auto location = locate_lateral_cell( corners, query.point, subdivision_level,
                        physical_wedge ? &coarse_radius : nullptr );
                    if ( !location.valid() )
                        continue;
                    query.diamond_id = diamond;
                    query.subdomain_x = location.x;
                    query.subdomain_y = location.y;
                    const T lower = physical_wedge ? radius - tolerance : radius;
                    const T upper = Kokkos::max( radius, coarse_radius ) + tolerance;
                    int lo = 0, hi = radial_domains;
                    while ( hi - lo > 1 )
                    {
                        const int mid = lo + ( hi - lo ) / 2;
                        if ( lower >= radial_bounds( mid ) ) lo = mid;
                        else hi = mid;
                    }
                    query.subdomain_r = lo;
                    if ( physical_wedge && lo + 1 < radial_domains && upper >= radial_bounds( lo + 1 ) )
                    {
                        const int slot = Kokkos::atomic_fetch_add( &radial_boundary_count(), 1 );
                        radial_boundary_queries( slot ) = i;
                    }
                    else
                        query.local_subdomain = local_lookup( query );
                    return;
                }
                Kokkos::atomic_add( &errors(), 1 );
            } );
        int radial_boundary_count_host = 0;
        if ( physical_wedge )
            Kokkos::deep_copy( radial_boundary_count_host, radial_boundary_count );
        if ( radial_boundary_count_host > 0 )
        {
            // Preserve exact physical ownership at radial interfaces, including split radial ranks.
            // Compaction keeps this expensive path out of ordinary routing launches.
            Kokkos::parallel_for(
                "point_queries_resolve_radial_boundary", radial_boundary_count_host,
                KOKKOS_LAMBDA( int i ) {
                    auto& query = outgoing( radial_boundary_queries( i ) );
                    dense::Vec< T, 3 > corners[4];
                    for ( int c = 0; c < 4; ++c )
                        for ( int d = 0; d < 3; ++d )
                            corners[c]( d ) = diamond_corners( query.diamond_id, c % 2, c / 2, d );
                    const int local_cells = n - 1;
                    const int diamond_cells = local_cells * local_lookup.lateral;
                    const RoutingCoordinates routing_coords{
                        coords, local_lookup,
                        { { corners[0], corners[1], corners[2], corners[3] }, diamond_cells },
                        query.diamond_id, local_cells };
                    const int radial_domains = radial_bounds.extent( 0 ) - 1;
                    T radius{};
                    const auto location = locate_diamond_cell_direct( routing_coords, diamond_cells, query.point,
                        radial_bounds( 0 ), radial_bounds( radial_domains ), radius );
                    if ( !location.valid() )
                    {
                        Kokkos::atomic_add( &errors(), 1 );
                        return;
                    }
                    query.subdomain_x = location.x / local_cells;
                    query.subdomain_y = location.y / local_cells;
                    radius = Kokkos::clamp( radius, radial_bounds( 0 ), radial_bounds( radial_domains ) );
                    int lo = 0, hi = radial_domains;
                    while ( hi - lo > 1 )
                    {
                        const int mid = lo + ( hi - lo ) / 2;
                        if ( radius >= radial_bounds( mid ) ) lo = mid;
                        else hi = mid;
                    }
                    query.subdomain_r = lo;
                    query.local_subdomain = local_lookup( query );
                } );
        }
        check_kernel_errors();
        const int rank = mpi::rank( domain_.comm() );
        const int ranks = mpi::num_processes( domain_.comm() );
        if ( remote_.extent( 0 ) < std::size_t( count ) )
            remote_ = Kokkos::View< Query* >( "remote_point_queries", count );
        const auto remote = remote_;
        const auto remote_count = remote_count_;
        Kokkos::deep_copy( remote_count, 0 );
        Kokkos::parallel_for( "mmoc_prepare_owner_samples", count, KOKKOS_LAMBDA( int i ) {
            const auto query = outgoing( i );
            if ( query.local_subdomain < 0 )
            {
                remote( Kokkos::atomic_fetch_add( &remote_count(), 1 ) ) = query;
                return;
            }
            const auto found = locate_owned( coords, radii, query.local_subdomain, query.point,
                radial_bounds( 0 ), radial_bounds( radial_bounds.extent( 0 ) - 1 ) );
            if ( found.subdomain < 0 ) Kokkos::atomic_add( &errors(), 1 );
            else locations( query.origin_node ) = found;
        } );
        check_kernel_errors();
        evaluate( locations, node_count_, output );
        Kokkos::deep_copy( count, remote_count );
        last_remote_queries_ = count;
        int global_remote_count = count;
        if ( ranks > 1 )
            MPI_Allreduce( MPI_IN_PLACE, &global_remote_count, 1, MPI_INT, MPI_MAX, domain_.comm() );
        if ( global_remote_count == 0 ) return;
        require( ranks > 1, "unresolved owner on a single rank" );
        Kokkos::Profiling::pushRegion( "mmoc_host_grouping" );
        if ( outgoing_host_.extent( 0 ) < std::size_t( count ) )
            outgoing_host_ = decltype( outgoing_host_ )( "mmoc_outgoing_host", count );
        auto host = Kokkos::subview( outgoing_host_, std::make_pair( 0, count ) );
        Kokkos::deep_copy( host, Kokkos::subview( remote, std::make_pair( 0, count ) ) );
        auto& batches = batches_;
        for ( auto& [peer, batch] : batches ) batch.clear();
        for ( int i = 0; i < count; ++i )
        {
            const auto& query = host( i );
            auto [owner, inserted] = destination_ranks_.try_emplace( local_lookup.key( query ), -1 );
            if ( inserted ) owner->second = subdomain_to_rank_(
                SubdomainInfo( query.diamond_id, query.subdomain_x, query.subdomain_y, query.subdomain_r ),
                local_lookup.lateral, local_lookup.radial );
            require( owner->second >= 0 && owner->second < ranks && owner->second != rank,
                     "MMOC owner mapping disagrees with the domain" );
            batches[owner->second].push_back( query );
        }
        Kokkos::Profiling::popRegion();
        const auto returned = communication::shell::exchange_point_queries< Query, Reply< Value > >(
            domain_.comm(), batches, [&]( const std::vector< Query >& received ) {
                const int received_count = communication::shell::detail::point_query_count( received.size(), domain_.comm() );
                if ( incoming_.extent( 0 ) < received.size() ) incoming_ = decltype( incoming_ )( "mmoc_incoming", received.size() );
                if ( locations_.extent( 0 ) < received.size() ) locations_ = decltype( locations_ )( "mmoc_locations", received.size() );
                auto q = Kokkos::subview( incoming_, std::make_pair( 0, received_count ) );
                Kokkos::Profiling::ScopedRegion receive_region( "mmoc_receiver_evaluation" );
                if ( incoming_host_.extent( 0 ) < received.size() )
                    incoming_host_ = decltype( incoming_host_ )( "mmoc_incoming_host", received.size() );
                auto q_host = Kokkos::subview( incoming_host_, std::make_pair( 0, received_count ) );
                for ( int i = 0; i < received_count; ++i ) q_host( i ) = received[i];
                Kokkos::deep_copy( q, q_host );
                const auto work = locations_;
                Kokkos::parallel_for( "mmoc_locate_received", received_count, KOKKOS_LAMBDA( int i ) {
                    const int s = local_lookup( q( i ) );
                    work( i ).subdomain = -1;
                    if ( s < 0 ) { Kokkos::atomic_add( &errors(), 1 ); return; }
                    work( i ) = locate_owned( coords, radii, s, q( i ).point, radial_bounds( 0 ),
                                             radial_bounds( radial_bounds.extent( 0 ) - 1 ) );
                    if ( work( i ).subdomain < 0 ) Kokkos::atomic_add( &errors(), 1 );
                } );
                check_kernel_errors();
                auto& storage = [&]() -> auto& {
                    if constexpr ( std::is_same_v< Value, T > ) return scalar_values_;
                    else return vector_values_;
                }();
                if ( storage.extent( 0 ) < received.size() )
                    storage = std::remove_reference_t< decltype( storage ) >( "mmoc_received_values", received.size() );
                evaluate( work, received_count, storage );
                auto values = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{},
                    Kokkos::subview( storage, std::make_pair( 0, received_count ) ) );
                std::vector< Reply< Value > > replies( received_count );
                for ( int i = 0; i < received_count; ++i ) replies[i] = { received[i].origin_node, values( i ) };
                return replies;
            } );
        require( returned.size() == std::size_t( count ), "missing MMOC query replies" );
        // Reuse a typed device reply buffer between calls; retain origin indices for scattering.
        auto& reply_storage = [&]() -> auto& {
            if constexpr ( std::is_same_v< Value, T > ) return scalar_replies_;
            else return vector_replies_;
        }();
        if ( reply_storage.extent( 0 ) < returned.size() )
            reply_storage = std::remove_reference_t< decltype( reply_storage ) >( "mmoc_returned", returned.size() );
        auto replies = Kokkos::subview( reply_storage, std::make_pair( 0, count ) );
        auto replies_host = Kokkos::create_mirror_view( replies );
        for ( std::size_t i = 0; i < returned.size(); ++i ) replies_host( i ) = returned[i];
        Kokkos::deep_copy( replies, replies_host );
        Kokkos::parallel_for( "mmoc_scatter_replies", count, KOKKOS_LAMBDA( int i ) {
            output( replies( i ).origin_node ) = replies( i ).value;
        } );
        Kokkos::fence();
    }
};
} // namespace terra::grid::shell
