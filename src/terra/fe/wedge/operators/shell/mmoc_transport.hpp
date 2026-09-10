#pragma once

#include <array>
#include <limits>
#include <vector>
#include <memory>
#include "terra/grid/shell/mmoc_point_sampling.hpp"
#include "terra/communication/shell/communication_plan.hpp"

#include "communication/shell/communication.hpp"
#include "fe/wedge/sl/ghost_exchange.hpp"
#include "fe/wedge/sl/ghosted_geometry.hpp"
#include "fe/wedge/sl/point_location.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "kokkos/kokkos_wrapper.hpp"
#include "linalg/vector_q1.hpp"
#include "util/timer.hpp"

/// @file
/// MMOC transport with owner-routed sampling at every Runge-Kutta stage.
/// Temperature uses monotone cubic reconstruction; velocity uses physical-wedge Q1 interpolation.
/// Ghost values extend interpolation stencils of owned cells. A point in a ghost cell is evaluated by
/// its owning subdomain, using bundled MPI queries when that subdomain is remote.
///
/// Backward tracing blends velocities with global pseudo-time tau=(substep+stage_weight)/substeps.
/// Query location and binning precede tiled interpolation, where teams share stencil data in scratch.
/// Diffusion remains a separate operator-splitting step in the energy solver.
///
/// The cubic method retains MMOC's index-space reconstruction and limiters. Its stencil stays within
/// one diamond: crossing a diamond seam would mix different index charts. Internal subdomain and radial
/// ghosts are usable; extrapolated radii beyond physical boundaries are excluded.

namespace terra::fe::wedge::operators::shell
{

/// @brief Explicit Runge-Kutta scheme used to trace the characteristics.
enum class TimeSteppingScheme
{
    /// First order.
    ExplicitEuler,
    /// Third order.
    RK3,
    /// Third order, Ralston's minimum-error coefficients.
    Ralston,
    /// Fourth order.
    RK4,
};

/// @brief Butcher tableau, sized for the schemes in \ref TimeSteppingScheme.
template < typename ScalarType >
struct ButcherTableau
{
    static constexpr int max_stages = 4;

    int        stages                       = 1;
    ScalarType A[max_stages][max_stages]    = {};
    ScalarType b[max_stages]                = {};
    ScalarType c[max_stages]                = {};
};

template < typename ScalarType >
ButcherTableau< ScalarType > butcher_tableau( const TimeSteppingScheme scheme )
{
    ButcherTableau< ScalarType > t;

    switch ( scheme )
    {
    case TimeSteppingScheme::ExplicitEuler:
        t.stages = 1;
        t.b[0]   = 1;
        t.c[0]   = 0;
        break;

    case TimeSteppingScheme::RK3:
        t.stages  = 3;
        t.A[1][0] = ScalarType( 0.5 );
        t.A[2][0] = ScalarType( -1 );
        t.A[2][1] = ScalarType( 2 );
        t.b[0]    = ScalarType( 1 ) / 6;
        t.b[1]    = ScalarType( 2 ) / 3;
        t.b[2]    = ScalarType( 1 ) / 6;
        t.c[0]    = 0;
        t.c[1]    = ScalarType( 0.5 );
        t.c[2]    = 1;
        break;

    case TimeSteppingScheme::Ralston:
        t.stages  = 3;
        t.A[1][0] = ScalarType( 0.5 );
        t.A[2][1] = ScalarType( 0.75 );
        t.b[0]    = ScalarType( 2 ) / 9;
        t.b[1]    = ScalarType( 1 ) / 3;
        t.b[2]    = ScalarType( 4 ) / 9;
        t.c[0]    = 0;
        t.c[1]    = ScalarType( 0.5 );
        t.c[2]    = ScalarType( 0.75 );
        break;

    case TimeSteppingScheme::RK4:
        t.stages  = 4;
        t.A[1][0] = ScalarType( 0.5 );
        t.A[2][1] = ScalarType( 0.5 );
        t.A[3][2] = ScalarType( 1 );
        t.b[0]    = ScalarType( 1 ) / 6;
        t.b[1]    = ScalarType( 1 ) / 3;
        t.b[2]    = ScalarType( 1 ) / 3;
        t.b[3]    = ScalarType( 1 ) / 6;
        t.c[0]    = 0;
        t.c[1]    = ScalarType( 0.5 );
        t.c[2]    = ScalarType( 0.5 );
        t.c[3]    = 1;
        break;
    }

    return t;
}

/// @brief Semi-Lagrangian (MMOC) advection of a Q1 nodal scalar field.
template < typename ScalarType >
class MMOCTransport
{
  public:
    using Vec3 = dense::Vec< ScalarType, 3 >;

    MMOCTransport(
        const grid::shell::DistributedDomain&       domain,
        const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >& ownership_mask,
        const TimeSteppingScheme                    scheme = TimeSteppingScheme::RK4,
        grid::shell::SubdomainToRankDistributionFunction owner = grid::shell::subdomain_to_rank_iterate_diamond_subdomains )
    : domain_( &domain )
    , ownership_mask_( ownership_mask )
    , exchange_( domain )
    , tableau_( butcher_tableau< ScalarType >( scheme ) )
    {
        coords_g_       = sl::ghosted_unit_sphere_coords< ScalarType >( domain, exchange_ );
        radii_g_        = sl::ghosted_shell_radii< ScalarType >( domain, exchange_ );
        lateral_valid_  = sl::ghosted_lateral_validity< ScalarType >( exchange_, coords_g_ );

        const auto bounds = sl::shell_radius_bounds< ScalarType >( domain );
        r_min_            = bounds.first;
        r_max_            = bounds.second;

        T_g_     = exchange_.allocate_scalar< ScalarType >( "mmoc_T_ghosted" );
        u_g_     = exchange_.allocate_vec< ScalarType, 3 >( "mmoc_u_ghosted" );
        u_old_g_ = exchange_.allocate_vec< ScalarType, 3 >( "mmoc_u_old_ghosted" );

        T_new_ = grid::Grid4DDataScalar< ScalarType >(
            "mmoc_T_new",
            exchange_.num_subdomains(),
            exchange_.num_nodes_lateral(),
            exchange_.num_nodes_lateral(),
            exchange_.num_nodes_radial() );
        const auto coords = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
        const auto radii = grid::shell::subdomain_shell_radii< ScalarType >( domain );
        sampler_ = std::make_unique< grid::shell::MMOCPointSampler< ScalarType > >(
            domain, coords, radii, ownership_mask, std::move( owner ) );
        interpolation_ = std::make_unique< sl::TiledInterpolator< ScalarType > >(
            domain, T_g_, u_g_, u_old_g_, radii_g_, lateral_valid_ );
        halo_ = std::make_unique< HaloPlan >( domain, true, true );
        halo_buffers_ = std::make_unique< communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarType > >( domain );
        const auto count = ownership_mask.size();
        trajectory_ = decltype( trajectory_ )( "mmoc_trajectory", count );
        positions_ = decltype( positions_ )( "mmoc_stage_positions", count );
        velocity_values_ = decltype( velocity_values_ )( "mmoc_stage_velocity", count );
        temperature_values_ = decltype( temperature_values_ )( "mmoc_temperature_values", count );
        stages_ = decltype( stages_ )( "mmoc_rk_stages", count );
    }

    /// Compatibility query: owner routing imposes no ghost-width Courant limit.
    [[nodiscard]] static constexpr ScalarType max_courant()
    {
        return std::numeric_limits< ScalarType >::infinity();
    }

    /// @brief Substeps giving a trajectory error comparable to the interpolation error.
    ///
    /// Purely an accuracy control. One substep is
    /// enough for a Courant number below one with a fourth-order scheme; the count is raised only so that the
    /// per-substep rotation angle stays small when the flow turns sharply within a timestep.
    [[nodiscard]] static int substeps_for_accuracy( const ScalarType courant )
    {
        const int n = static_cast< int >( Kokkos::ceil( courant ) );
        return n < 1 ? 1 : n;
    }

    /// @brief Advances `T` from t^n to t^{n+1} along the characteristics of the given velocity fields.
    ///
    /// @param T             in/out, the transported field
    /// @param u             velocity at t^{n+1}
    /// @param u_old         velocity at t^n
    /// @param dt            the full timestep
    /// @param substeps      number of substeps; see \ref substeps_for_accuracy
    /// @param global_limiter clip the interpolated value to the global range of T^n
    void step(
        linalg::VectorQ1Scalar< ScalarType >&          T,
        const linalg::VectorQ1Vec< ScalarType, 3 >&    u,
        const linalg::VectorQ1Vec< ScalarType, 3 >&    u_old,
        const ScalarType                               dt,
        const int                                      substeps,
        const bool                                     global_limiter = true )
    {
        util::Timer timer( "mmoc_transport" );

        Kokkos::Profiling::ScopedRegion transport_region( "mmoc_transport" );
        {
            Kokkos::Profiling::ScopedRegion ghost_region( "mmoc_ghost_fill" );
            exchange_.fill( T.grid_data(), T_g_ );
            // Owner routing guarantees that Q1 velocity sampling uses only owned-cell vertices.
            // Populate the existing padded layout without exchanging its unused ghost layers.
            exchange_.copy_interior( u.grid_data(), u_g_ );
            exchange_.copy_interior( u_old.grid_data(), u_old_g_ );
        }

        ScalarType t_min = std::numeric_limits< ScalarType >::max();
        ScalarType t_max = std::numeric_limits< ScalarType >::lowest();

        if ( global_limiter )
        {
            const auto T_data = T.grid_data();
            Kokkos::parallel_reduce(
                "mmoc_global_range",
                grid::shell::local_domain_md_range_policy_nodes( *domain_ ),
                KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r, ScalarType& lo,
                               ScalarType& hi ) {
                    lo = Kokkos::min( lo, T_data( sd, x, y, r ) );
                    hi = Kokkos::max( hi, T_data( sd, x, y, r ) );
                },
                Kokkos::Min< ScalarType >( t_min ),
                Kokkos::Max< ScalarType >( t_max ) );
            Kokkos::fence();

            MPI_Allreduce( MPI_IN_PLACE, &t_min, 1, MPI_DOUBLE, MPI_MIN, domain_->comm() );
            MPI_Allreduce( MPI_IN_PLACE, &t_max, 1, MPI_DOUBLE, MPI_MAX, domain_->comm() );
        }
        else
        {
            t_min = std::numeric_limits< ScalarType >::lowest();
            t_max = std::numeric_limits< ScalarType >::max();
        }

        communication::shell::detail::point_query_require( substeps > 0 && std::isfinite( dt ), domain_->comm(),
                                                          "invalid MMOC timestep or substep count" );
        last_remote_queries_ = 0;

        long long escapes = 0;
        trace( T, dt, substeps, t_min, t_max, escapes );

        Kokkos::deep_copy( T.grid_data(), T_new_ );

        // Only owning nodes are evaluated; SUM distributes their values to shared copies.
        {
            Kokkos::Profiling::ScopedRegion halo_region( "mmoc_halo" );
            halo_->exchange_and_reduce( T.grid_data(), *halo_buffers_ );
        }

        MPI_Allreduce( MPI_IN_PLACE, &escapes, 1, MPI_LONG_LONG, MPI_SUM, domain_->comm() );
        last_escapes_ = escapes;
    }

    /// Compatibility diagnostics. Successful steps resolve every sample; lookup errors abort the communicator.
    [[nodiscard]] std::vector< std::array< int, 4 > > last_escape_locations() const { return {}; }
    [[nodiscard]] long long last_escapes() const { return last_escapes_; }

    /// @internal Traces the characteristics and writes the result into `T_new_`.
    ///
    /// Public only because CUDA does not permit an extended `__host__ __device__` lambda inside a private or
    /// protected member function. Not part of the interface -- use \ref step.
    void trace(
        const linalg::VectorQ1Scalar< ScalarType >& /*T*/,
        const ScalarType                            dt,
        const int                                   substeps,
        const ScalarType                            t_min,
        const ScalarType                            t_max,
        long long&                                  escapes )
    {
        const auto coords = coords_g_;
        const auto radii = radii_g_;
        const auto ownership = ownership_mask_;
        const auto trajectory = trajectory_, positions = positions_, velocities = velocity_values_;
        const auto temperatures = temperature_values_;
        const auto stages = stages_;
        const auto output = T_new_;
        const auto tableau = tableau_;
        const int n = ownership.extent( 1 ), nr = ownership.extent( 3 );
        const int count = communication::shell::detail::point_query_count( ownership.size(), domain_->comm() );
        const ScalarType h = dt / ScalarType( substeps );
        Kokkos::parallel_for( "mmoc_initialize_trajectories", count, KOKKOS_LAMBDA( int i ) {
            const int s = i / ( n * n * nr ), x = ( i / ( n * nr ) ) % n, y = ( i / nr ) % n, r = i % nr;
            for ( int d = 0; d < 3; ++d )
                trajectory( i )( d ) = coords( s, x + sl::ghost_width, y + sl::ghost_width, r + sl::ghost_width, d ) *
                                      radii( s, r + sl::ghost_width );
        } );
        for ( int m = 0; m < substeps; ++m )
        {
            for ( int stage = 0; stage < tableau.stages; ++stage )
            {
                Kokkos::parallel_for( "mmoc_rk_stage_positions", count, KOKKOS_LAMBDA( int i ) {
                    auto point = trajectory( i );
                    for ( int j = 0; j < stage; ++j ) point = point + stages( i, j ) * ( h * tableau.A[stage][j] );
                    positions( i ) = point;
                } );
                const ScalarType tau = ( ScalarType( m ) + tableau.c[stage] ) / ScalarType( substeps );
                sampler_->sample( positions, velocities, [&]( const auto& work, int size, const auto& values ) {
                    interpolation_->template evaluate< true >( work, size, values, tau, shared_interpolation_ );
                } );
                last_remote_queries_ += sampler_->last_remote_queries();
                Kokkos::parallel_for( "mmoc_rk_store_stage", count, KOKKOS_LAMBDA( int i ) {
                    stages( i, stage ) = velocities( i ) * ScalarType( -1 );
                } );
            }
            Kokkos::parallel_for( "mmoc_rk_advance", count, KOKKOS_LAMBDA( int i ) {
                auto point = trajectory( i );
                for ( int stage = 0; stage < tableau.stages; ++stage ) point = point + stages( i, stage ) * ( h * tableau.b[stage] );
                trajectory( i ) = point;
            } );
        }
        sampler_->sample( trajectory, temperatures, [&]( const auto& work, int size, const auto& values ) {
            interpolation_->template evaluate< false >( work, size, values, ScalarType( 0 ), shared_interpolation_ );
        } );
        last_remote_queries_ += sampler_->last_remote_queries();
        Kokkos::deep_copy( output, ScalarType( 0 ) );
        Kokkos::parallel_for( "mmoc_store_temperature", count, KOKKOS_LAMBDA( int i ) {
            const int s = i / ( n * n * nr ), x = ( i / ( n * nr ) ) % n, y = ( i / nr ) % n, r = i % nr;
            if ( util::has_flag( ownership( s, x, y, r ), grid::NodeOwnershipFlag::OWNED ) )
                output( s, x, y, r ) = Kokkos::clamp( temperatures( i ), t_min, t_max );
        } );
        Kokkos::fence();
        escapes = 0;
    }

    // Diagnostic reference execution uses identical stencils and arithmetic without team scratch.
    void set_shared_interpolation( bool enabled ) { shared_interpolation_ = enabled; }
    std::size_t last_remote_queries() const { return last_remote_queries_; }
    std::size_t sampling_buffer_bytes() const
    {
        return sampler_->buffer_bytes() + interpolation_->buffer_bytes() +
            ( trajectory_.span() + positions_.span() + velocity_values_.span() + stages_.span() ) * sizeof( Vec3 ) +
            temperature_values_.span() * sizeof( ScalarType );
    }

  private:
    const grid::shell::DistributedDomain*                    domain_ = nullptr;
    grid::Grid4DDataScalar< grid::NodeOwnershipFlag >        ownership_mask_;
    sl::GhostExchange                                        exchange_;
    ButcherTableau< ScalarType >                             tableau_;

    grid::Grid4DDataVec< ScalarType, 3 >    coords_g_;
    grid::Grid2DDataScalar< ScalarType >    radii_g_;
    Kokkos::View< uint8_t*** >              lateral_valid_;
    ScalarType                              r_min_ = 0;
    ScalarType                              r_max_ = 0;

    grid::Grid4DDataScalar< ScalarType > T_g_;
    grid::Grid4DDataVec< ScalarType, 3 > u_g_;
    grid::Grid4DDataVec< ScalarType, 3 > u_old_g_;
    grid::Grid4DDataScalar< ScalarType > T_new_;

    long long last_escapes_ = 0;

    using HaloPlan = communication::shell::ShellBoundaryCommPlan< grid::Grid4DDataScalar< ScalarType > >;
    std::unique_ptr< grid::shell::MMOCPointSampler< ScalarType > > sampler_;
    std::unique_ptr< sl::TiledInterpolator< ScalarType > > interpolation_;
    std::unique_ptr< HaloPlan > halo_;
    std::unique_ptr< communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarType > > halo_buffers_;
    Kokkos::View< Vec3* > trajectory_, positions_, velocity_values_;
    Kokkos::View< Vec3*[4] > stages_;
    Kokkos::View< ScalarType* > temperature_values_;
    bool shared_interpolation_ = true;
    std::size_t last_remote_queries_ = 0;
};

} // namespace terra::fe::wedge::operators::shell
