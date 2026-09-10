#pragma once

#include <array>
#include <limits>
#include <vector>

#include "communication/shell/communication.hpp"
#include "fe/wedge/sl/ghost_exchange.hpp"
#include "fe/wedge/sl/ghosted_geometry.hpp"
#include "fe/wedge/sl/point_location.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "kokkos/kokkos_wrapper.hpp"
#include "linalg/vector_q1.hpp"
#include "util/timer.hpp"

/// @file
///
/// Modified method of characteristics (MMOC) transport of a Q1 nodal field on the spherical shell.
///
/// Solves the pure advection problem
/// \f[
///   \partial_t T + \mathbf{u} \cdot \nabla T = 0, \qquad t \in [t^n, t^{n+1}]
/// \f]
/// exactly along characteristics: for every node \f$ x_i \f$ the characteristic is traced **backwards** by
/// integrating \f$ \mathrm{d}X/\mathrm{d}s = -\hat{\mathbf{u}}(X, s) \f$ from \f$ X(0) = x_i \f$ over
/// \f$ s \in [0, \Delta t] \f$, and the new value is the old field evaluated at the foot point,
/// \f$ T^{n+1}_i = T^n(X(\Delta t)) \f$.
///
/// The velocity is interpolated linearly in time between the two given fields. Because the integration runs
/// backwards, pseudo-time \f$ s \f$ maps to physical time \f$ t^{n+1} - s \f$, so a stage at
/// \f$ \tau = s / \Delta t \f$ uses \f$ (1 - \tau) \mathbf{u}^{n+1} + \tau \mathbf{u}^{n} \f$.
///
/// @note With more than one substep the interpolation weight is tracked in *global* pseudo-time,
///       \f$ \tau = (m + c_k) / M \f$ for substep \f$ m \f$ of \f$ M \f$ and stage \f$ k \f$. Reusing the raw
///       stage weight \f$ c_k \f$ in every substep -- as some implementations do -- would interpolate as if each
///       substep spanned the whole timestep and silently drop the temporal order back to first order.
///
/// The scheme is unconditionally stable in the advective sense: there is no CFL restriction from the
/// characteristic tracing itself. The *implementation* is bounded by the ghost layer width, since a foot point
/// must remain inside the local subdomain plus its ghost layer. \ref MMOCTransport::recommended_substeps
/// converts a Courant number into the number of substeps that keeps every substep inside that budget.
///
/// **Interpolation.** The field at the foot point is evaluated with
/// \ref terra::fe::wedge::sl::evaluate_cubic_scalar — a monotone cubic reconstruction over the structured
/// index stencil — rather than with the Q1 shape functions of the containing wedge, because multilinear
/// evaluation is only second order and pays that error once per timestep, which is the dominant source of
/// numerical diffusion here. The velocity stays on \ref terra::fe::wedge::sl::evaluate_q1_vec; raising it made
/// no measurable difference to the rotation test and costs an order of magnitude more.
///
/// @warning This trade is **not yet settled**, and the choice of evaluator is deliberately left visible in
///          \ref trace so it can be flipped back. Measured on `test_mmoc_rotation` at level 5, one full
///          revolution of the cone: the cubic retains far more of the peak (0.64 versus 0.13 of the exact 1.0,
///          i.e. much less numerical diffusion, which is what it was introduced for) but its L2 error against
///          the exact solution is worse (1.79 versus 0.90). The reason shows up in the linear-field test: a
///          field \f$ a + b z \f$ is reproduced by the Q1 wedge evaluation to round-off — the Q1 map *is* the
///          geometric map, so it is exact for anything linear in \f$ x \f$ — whereas the cubic converges on it
///          at only about order 1.4 (3.6e-2, 1.4e-2, 5.2e-3 at levels 3, 4, 5). A tensor-product stencil in
///          index space presumes the field is a smooth function of the indices, and on this grid it is not
///          smooth enough: the map kinks where a stencil crosses into a neighbouring diamond (confining the
///          stencil to the owned block, as it now is, already recovered a factor of ten) and the recursive
///          bisection places nodes off any smooth parametrisation at \f$ \mathcal{O}(h^2) \f$. Neither
///          limiter is implicated — both the PCHIP sweep and the local range clip are inactive in these runs.
///          Recovering the order properly needs a reconstruction that is exact for polynomials in the physical
///          coordinates, e.g. a least-squares fit over the stencil in a local tangent frame with the
///          coefficients precomputed per cell.
///
///
/// Diffusion is not part of this operator; combine it with an implicit diffusion solve by operator splitting.

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
        const TimeSteppingScheme                    scheme = TimeSteppingScheme::RK4 )
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
    }

    /// @brief Largest Courant number this implementation supports for a single \ref step.
    ///
    /// The characteristic tracing itself is unconditionally stable -- there is no CFL condition in the scheme.
    /// What is bounded is the *distance* the departure point may travel: it has to stay inside the local
    /// subdomain plus its ghost layer, because that is all the data available to interpolate from. Substepping
    /// does **not** relax this: substeps refine the trajectory but the foot point still ends up a full Courant
    /// number away. Raising the limit means widening \ref terra::fe::wedge::sl::ghost_width.
    [[nodiscard]] static constexpr ScalarType max_courant()
    {
        // A margin below the ghost width, so that the foot point stays strictly inside the ghosted region.
        return ScalarType( 0.9 ) * sl::ghost_width;
    }

    /// @brief Substeps giving a trajectory error comparable to the interpolation error.
    ///
    /// Purely an accuracy control -- see \ref max_courant for what actually limits the timestep. One substep is
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
    /// @param substeps      number of substeps; see \ref recommended_substeps
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

        exchange_.fill( T.grid_data(), T_g_ );
        exchange_.fill( u.grid_data(), u_g_ );
        exchange_.fill( u_old.grid_data(), u_old_g_ );

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

        Kokkos::deep_copy( num_escape_locations_, 0 );

        long long escapes = 0;
        trace( T, dt, substeps, t_min, t_max, escapes );

        Kokkos::deep_copy( T.grid_data(), T_new_ );

        // Duplicated interface nodes are computed independently on each side from identical ghost data; a MAX
        // reduction removes any residual tie-breaking difference and keeps the field single-valued.
        communication::shell::send_recv( *domain_, T.grid_data(), communication::CommunicationReduction::MAX );

        MPI_Allreduce( MPI_IN_PLACE, &escapes, 1, MPI_LONG_LONG, MPI_SUM, domain_->comm() );
        last_escapes_ = escapes;
    }

    /// @brief Indices of up to 16 nodes that escaped in the last \ref step, as (subdomain, x, y, r).
    ///
    /// Diagnostic only; the count in \ref last_escapes is authoritative.
    [[nodiscard]] std::vector< std::array< int, 4 > > last_escape_locations() const
    {
        auto host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, escape_locations_ );
        auto n    = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, num_escape_locations_ );

        std::vector< std::array< int, 4 > > result;
        for ( int i = 0; i < Kokkos::min( n(), max_escape_locations ); ++i )
            result.push_back( { host( i, 0 ), host( i, 1 ), host( i, 2 ), host( i, 3 ) } );
        return result;
    }

    /// @brief Number of nodes whose departure point could not be located in the last \ref step.
    ///
    /// Those nodes keep their previous value (no advection). A non-zero count means the per-substep Courant
    /// number exceeded the ghost layer width, or that the departure point fell into one of the degenerate
    /// corner regions at the twelve pentagonal points of the icosahedral grid.
    [[nodiscard]] long long last_escapes() const { return last_escapes_; }

    /// @internal Traces the characteristics and writes the result into `T_new_`.
    ///
    /// Public only because CUDA does not permit an extended `__host__ __device__` lambda inside a private or
    /// protected member function. Not part of the interface -- use \ref step.
    void trace(
        const linalg::VectorQ1Scalar< ScalarType >& T,
        const ScalarType                            dt,
        const int                                   substeps,
        const ScalarType                            t_min,
        const ScalarType                            t_max,
        long long&                                  escapes )
    {
        const sl::IndexBounds bounds{ exchange_.num_nodes_lateral_ghosted(),
                                      exchange_.num_nodes_lateral_ghosted(),
                                      exchange_.num_nodes_radial_ghosted() };

        const sl::RadialSliceCoords< decltype( coords_g_ ) > lateral{ coords_g_, sl::ghost_width };

        const auto coords_g = coords_g_;
        const auto radii_g  = radii_g_;
        const auto T_g      = T_g_;
        const auto u_g      = u_g_;
        const auto u_old_g  = u_old_g_;
        const auto T_new    = T_new_;
        const auto tableau  = tableau_;

        const int  n_lat_g     = exchange_.num_nodes_lateral_ghosted();
        const int  n_rad_g     = exchange_.num_nodes_radial_ghosted();
        const int  n_lat_owned = exchange_.num_nodes_lateral();
        const int  n_rad_owned = exchange_.num_nodes_radial();
        const auto r_min   = r_min_;
        const auto r_max   = r_max_;

        const ScalarType h        = dt / static_cast< ScalarType >( substeps );
        const ScalarType inv_M    = ScalarType( 1 ) / static_cast< ScalarType >( substeps );
        constexpr int    max_walk = 4 * sl::ghost_width + 4;
        constexpr auto   eps      = ScalarType( 1e-12 );

        const auto T_old         = T.grid_data();
        const auto lateral_valid = lateral_valid_;
        const auto escape_loc     = escape_locations_;
        const auto escape_loc_num = num_escape_locations_;

        Kokkos::parallel_reduce(
            "mmoc_trace_characteristics",
            grid::shell::local_domain_md_range_policy_nodes( *domain_ ),
            KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r, long long& esc ) {
                const int gx = sl::to_ghosted_index( x );
                const int gy = sl::to_ghosted_index( y );
                const int gr = sl::to_ghosted_index( r );

                Vec3 p;
                for ( int d = 0; d < 3; ++d )
                    p( d ) = coords_g( sd, gx, gy, gr, d );

                Vec3 X = p * radii_g( sd, gr );

                // Stencil the cubic interpolation may read from. Laterally it is the owned block only: a
                // lateral ghost row holds a neighbour's real values, but it belongs to another diamond, and
                // the index-space parametrisation kinks at that seam -- a stencil straddling it is
                // inconsistent, not just less accurate. Radially the ghost layers outside the CMB and the
                // surface exist only so that the radii array stays monotone; they hold extrapolated radii and
                // no field data at all, so trim them off too. In both directions the window then slides
                // inwards near the edge (one-sided but never extrapolating), and a foot point that lands
                // outside the range altogether falls back to the Q1 evaluation.
                sl::StencilBounds stencil{ { sl::ghost_width, n_lat_g - 1 - sl::ghost_width },
                                           { sl::ghost_width, n_lat_g - 1 - sl::ghost_width },
                                           { 0, n_rad_g - 1 } };
                {
                    const ScalarType r_tol = ScalarType( 1e-12 ) * r_max;
                    while ( stencil.r.lo < stencil.r.hi && radii_g( sd, stencil.r.lo ) < r_min - r_tol )
                        ++stencil.r.lo;
                    while ( stencil.r.hi > stencil.r.lo && radii_g( sd, stencil.r.hi ) > r_max + r_tol )
                        --stencil.r.hi;
                }

                // Seed with a cell all of whose nodes are owned: it always contains this node as a vertex,
                // and it can never be one of the degenerate ghost-corner wedges.
                sl::WedgeCell cell{ sl::to_ghosted_index( Kokkos::min( x, n_lat_owned - 2 ) ),
                                    sl::to_ghosted_index( Kokkos::min( y, n_lat_owned - 2 ) ),
                                    sl::to_ghosted_index( Kokkos::min( r, n_rad_owned - 2 ) ),
                                    0 };

                bool escaped = false;

                for ( int m = 0; m < substeps && !escaped; ++m )
                {
                    Vec3 kv[ButcherTableau< ScalarType >::max_stages];
                    const Vec3 X_start = X;

                    for ( int s = 0; s < tableau.stages; ++s )
                    {
                        Vec3 Y = X_start;
                        for ( int l = 0; l < s; ++l )
                            Y = Y + kv[l] * ( h * tableau.A[s][l] );

                        const auto res = sl::locate_point(
                            Y, sd, cell, lateral, radii_g, bounds, max_walk, eps,
                            /*clamp_radially=*/true, r_min, r_max, lateral_valid );

                        if ( !res.found )
                        {
                            escaped = true;
                            break;
                        }
                        cell = res.cell;

                        // The velocity is smooth by construction, so it is interpolated with the unlimited
                        // cubic: an error here displaces the foot point and enters T just as directly as an
                        // error in T itself, and a limiter could only cost accuracy.
                        const auto u_new_s =
                            sl::evaluate_q1_vec< ScalarType, 3 >( u_g, sd, res.cell, res.xi, res.eta, res.zeta );
                        const auto u_old_s = sl::evaluate_q1_vec< ScalarType, 3 >(
                            u_old_g, sd, res.cell, res.xi, res.eta, res.zeta );

                        // Global pseudo-time of this stage, in [0, 1] across the whole timestep.
                        const ScalarType tau = ( static_cast< ScalarType >( m ) + tableau.c[s] ) * inv_M;

                        for ( int d = 0; d < 3; ++d )
                            kv[s]( d ) = -( ( ScalarType( 1 ) - tau ) * u_new_s( d ) + tau * u_old_s( d ) );
                    }

                    if ( escaped )
                        break;

                    Vec3 X_next = X_start;
                    for ( int s = 0; s < tableau.stages; ++s )
                        X_next = X_next + kv[s] * ( h * tableau.b[s] );
                    X = X_next;
                }

                if ( !escaped )
                {
                    const auto res = sl::locate_point(
                        X, sd, cell, lateral, radii_g, bounds, max_walk, eps,
                        /*clamp_radially=*/true, r_min, r_max, lateral_valid );

                    if ( res.found )
                    {
                        const ScalarType value = sl::evaluate_cubic_scalar(
                            T_g, sd, res.cell, res.xi, res.eta, res.zeta, radii_g, stencil, lateral_valid,
                            /*monotone=*/true );
                        T_new( sd, x, y, r ) = Kokkos::clamp( value, t_min, t_max );
                        return;
                    }
                }

                // The departure point left the ghosted region, or the walk ran into one of the degenerate
                // corner wedges. `cell` still holds the last wedge that was accepted, so interpolate at the
                // point of that wedge closest to the departure point: a bounded, first-order error, rather
                // than leaving the node un-advected for a whole timestep.
                {
                    ScalarType xi = 0, eta = 0, zeta = 0;
                    sl::clamp_to_wedge( X, sd, cell, lateral, radii_g, xi, eta, zeta );
                    const ScalarType value = sl::evaluate_cubic_scalar(
                        T_g, sd, cell, xi, eta, zeta, radii_g, stencil, lateral_valid, /*monotone=*/true );
                    T_new( sd, x, y, r )   = Kokkos::clamp( value, t_min, t_max );
                }
                esc += 1;

                const int slot = Kokkos::atomic_fetch_add( &escape_loc_num(), 1 );
                if ( slot < max_escape_locations )
                {
                    escape_loc( slot, 0 ) = sd;
                    escape_loc( slot, 1 ) = x;
                    escape_loc( slot, 2 ) = y;
                    escape_loc( slot, 3 ) = r;
                }
            },
            escapes );
        Kokkos::fence();
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

    static constexpr int          max_escape_locations = 16;
    Kokkos::View< int* [4] >      escape_locations_{ "mmoc_escape_locations", max_escape_locations };
    Kokkos::View< int >           num_escape_locations_{ "mmoc_num_escape_locations" };
};

} // namespace terra::fe::wedge::operators::shell
