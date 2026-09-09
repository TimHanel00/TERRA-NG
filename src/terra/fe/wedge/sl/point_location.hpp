#pragma once

#include "dense/mat.hpp"
#include "dense/vec.hpp"
#include "grid/grid_types.hpp"
#include "kokkos/kokkos_wrapper.hpp"

#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"

/// @file
///
/// Point location and evaluation of Q1 wedge fields at arbitrary physical points.
///
/// This is the geometric core of the semi-Lagrangian (MMOC) transport scheme. It exploits the fact that the
/// shell mesh is a tensor product of a laterally triangulated sphere and a set of radial shells, so that the
/// wedge geometric map
///
/// \f[
///    x(\xi, \eta, \zeta) = r(\zeta) \cdot \big[ (1 - \xi - \eta) p_1 + \xi p_2 + \eta p_3 \big],
///    \qquad r(\zeta) = r_1 + \tfrac{1}{2} (r_2 - r_1)(1 + \zeta)
/// \f]
///
/// (see \ref terra::fe::wedge::forward_map) can be inverted in **closed form**: with
/// \f$ A = [p_1 \, p_2 \, p_3] \f$ and \f$ \mu = A^{-1} x \f$ we get \f$ \rho = \mu_1 + \mu_2 + \mu_3 \f$ and
/// the barycentric coordinates \f$ \lambda = \mu / \rho \f$. The point lies in the wedge iff all
/// \f$ \lambda_i \geq 0 \f$ and \f$ r_1 \leq \rho \leq r_2 \f$.
///
/// Note that \f$ \lambda \f$ and \f$ \rho \f$ are **independent of the radial cell index**: the lateral search
/// and the radial search fully decouple. We therefore first walk laterally until the barycentric coordinates
/// are non-negative, then locate the radial cell by a search over the (monotone) shell radii.
///
/// @note \f$ \rho \neq |x| \f$ in general: the flat triangle spanned by \f$ p_1, p_2, p_3 \f$ is inscribed in
///       the unit sphere, so \f$ \rho = |x| (1 + \mathcal{O}(h^2)) \f$. The radial search must use \f$ \rho \f$.

namespace terra::fe::wedge::sl {

/// @brief A wedge cell inside one subdomain's node index space.
///
/// `(x, y)` is the hex cell index (spanning nodes `x, x+1` and `y, y+1`), `r` the radial cell index (spanning
/// nodes `r` and `r+1`), and `w` selects one of the two triangles the hex cell is split into:
///
/// \code
///   2--3
///   |\ |     w = 0: (p1, p2, p3) = ( (x,y), (x+1,y), (x,y+1) )
///   | \|     w = 1: (p1, p2, p3) = ( (x+1,y+1), (x,y+1), (x+1,y) )
///   0--1
/// \endcode
///
/// This matches the ordering produced by \ref terra::fe::wedge::wedge_surface_physical_coords.
struct WedgeCell
{
    int x = 0;
    int y = 0;
    int r = 0;
    int w = 0;
};

/// @brief Lateral node indices of the three triangle vertices of a wedge cell.
KOKKOS_INLINE_FUNCTION void wedge_lateral_node_indices( const WedgeCell& c, int ( &nx )[3], int ( &ny )[3] )
{
    if ( c.w == 0 )
    {
        nx[0] = c.x;     ny[0] = c.y;
        nx[1] = c.x + 1; ny[1] = c.y;
        nx[2] = c.x;     ny[2] = c.y + 1;
    }
    else
    {
        nx[0] = c.x + 1; ny[0] = c.y + 1;
        nx[1] = c.x;     ny[1] = c.y + 1;
        nx[2] = c.x + 1; ny[2] = c.y;
    }
}

/// @brief Moves to the wedge cell sharing the edge opposite to local vertex `v` (0, 1 or 2).
///
/// Every edge crossing flips the triangle parity. From `w == 0` the hex cell offsets are
/// `(0,0)`, `(-1,0)`, `(0,-1)`; from `w == 1` they are `(0,0)`, `(+1,0)`, `(0,+1)`.
KOKKOS_INLINE_FUNCTION WedgeCell wedge_edge_neighbor( const WedgeCell& c, const int v )
{
    WedgeCell n = c;
    n.w         = 1 - c.w;

    if ( c.w == 0 )
    {
        // v == 0: opposite edge is the hex diagonal -> other triangle of the same hex cell.
        if ( v == 1 )
            n.x = c.x - 1;
        else if ( v == 2 )
            n.y = c.y - 1;
    }
    else
    {
        if ( v == 1 )
            n.x = c.x + 1;
        else if ( v == 2 )
            n.y = c.y + 1;
    }

    return n;
}

/// @brief Cone coordinates of a physical point w.r.t. the triangle of a wedge cell.
///
/// Solves \f$ A \mu = X \f$ with \f$ A = [p_1 \, p_2 \, p_3] \f$ (unit-sphere vertices).
///
/// The **unnormalised** \f$ \mu \f$ is what the point location works with: the cone spanned by the triangle
/// is exactly \f$ \{ x : A^{-1} x \geq 0 \} \f$, so `mu >= 0` componentwise is the containment test, and the
/// most negative component identifies the edge that was crossed. By Cramer's rule
/// \f$ \mu_i \propto \det( \ldots, X, \ldots ) \f$, i.e. the signs of \f$ \mu \f$ are exactly the
/// great-circle side tests of the spherical triangle.
///
/// @warning Do **not** normalise to \f$ \lambda = \mu / \rho \f$ before the containment test. For triangles
///          more than 90 degrees away from the direction of \f$ X \f$ the cone radius
///          \f$ \rho = \sum_i \mu_i \f$ is negative, and dividing flips all three signs — a point outside
///          the triangle then looks like it is inside. A single icosahedral diamond spans roughly 127 degrees,
///          so this happens well within one subdomain. Normalise only after the walk has converged.
///
/// The result depends only on the *lateral* indices of `cell`; `cell.r` is ignored.
template < typename T, typename CoordsShellType >
KOKKOS_INLINE_FUNCTION void wedge_lateral_cone_coords(
    const dense::Vec< T, 3 >& X,
    const int                 subdomain,
    const WedgeCell&          cell,
    const CoordsShellType&    coords_shell,
    dense::Vec< T, 3 >&       mu )
{
    int nx[3], ny[3];
    wedge_lateral_node_indices( cell, nx, ny );

    dense::Vec< T, 3 > p[3];
    for ( int v = 0; v < 3; ++v )
    {
        for ( int d = 0; d < 3; ++d )
        {
            p[v]( d ) = coords_shell( subdomain, nx[v], ny[v], d );
        }
    }

    const auto A = dense::Mat< T, 3, 3 >::from_col_vecs( p[0], p[1], p[2] );

    // mu is the unique coefficient vector with X = mu_1 p_1 + mu_2 p_2 + mu_3 p_3, so `mu >= 0` is the cone
    // containment test irrespective of how the triangle is wound.
    mu = A.inv() * X;
}

/// @brief Adapts a `(subdomain, x, y, r, d)` coordinate view to the lateral `(subdomain, x, y, d)` accessor
///        expected here, by pinning the radial index.
///
/// The unit-sphere node directions do not depend on the radial index, but the ghosted geometry stores them in a
/// four-dimensional (radially replicated) view so that the lateral ghost exchange can transport them. Pin any
/// owned radial layer.
template < typename ViewType >
struct RadialSliceCoords
{
    ViewType view;
    int      r = 0;

    KOKKOS_INLINE_FUNCTION auto operator()( const int subdomain, const int x, const int y, const int d ) const
    {
        return view( subdomain, x, y, r, d );
    }
};

/// @brief True if all three lateral nodes of a wedge cell carry usable geometry.
///
/// See \ref terra::fe::wedge::sl::ghosted_lateral_validity: a handful of diagonal ghost corners are degenerate
/// at the pentagonal points of the icosahedral grid, and the wedges touching them must not be used.
template < typename LateralValidityType >
KOKKOS_INLINE_FUNCTION bool wedge_lateral_nodes_valid(
    const WedgeCell&           cell,
    const int                  subdomain,
    const LateralValidityType& lateral_valid )
{
    int nx[3], ny[3];
    wedge_lateral_node_indices( cell, nx, ny );

    for ( int v = 0; v < 3; ++v )
    {
        if ( lateral_valid( subdomain, nx[v], ny[v] ) == 0 )
            return false;
    }
    return true;
}

/// @brief Bounds of the node index space a walk may visit.
///
/// For a plain (non-ghosted) subdomain field these are the subdomain node extents. For a ghosted field they
/// are the extents of the ghosted view, so that the walk may leave the owned region by up to the ghost width.
struct IndexBounds
{
    int num_nodes_x = 0;
    int num_nodes_y = 0;
    int num_nodes_r = 0;
};

/// @brief Outcome of a point location.
template < typename T >
struct LocateResult
{
    WedgeCell cell;
    T         xi   = 0;
    T         eta  = 0;
    T         zeta = 0;

    /// Point was located inside the index space.
    bool found = false;
    /// The lateral walk left the index space (departure point outside the local + ghost region).
    bool escaped_laterally = false;
    /// The point is radially outside the shell; `zeta` has been clamped to the outermost/innermost cell.
    bool clamped_radially = false;
    /// The walk reached a wedge touching a degenerate ghost node and stopped there.
    bool hit_invalid_node = false;
};

/// @brief Reference coordinates of the point of a wedge cell closest to `X`, for use when `X` itself could not
///        be located.
///
/// Projects onto the cell by clamping the barycentric coordinates onto the simplex and the cone radius onto the
/// cell's radial extent. Used as the fallback for a departure point that left the ghosted region: interpolating
/// at the nearest representable point is a bounded, first-order error, whereas skipping the advection entirely
/// leaves the node a whole timestep behind.
template < typename T, typename CoordsShellType, typename CoordsRadiiType >
KOKKOS_INLINE_FUNCTION void clamp_to_wedge(
    const dense::Vec< T, 3 >& X,
    const int                 subdomain,
    const WedgeCell&          cell,
    const CoordsShellType&    coords_shell,
    const CoordsRadiiType&    coords_radii,
    T&                        xi,
    T&                        eta,
    T&                        zeta )
{
    dense::Vec< T, 3 > mu;
    wedge_lateral_cone_coords( X, subdomain, cell, coords_shell, mu );

    T sum = T( 0 );
    for ( int v = 0; v < 3; ++v )
    {
        mu( v ) = Kokkos::max( mu( v ), T( 0 ) );
        sum += mu( v );
    }
    if ( sum <= T( 0 ) )
    {
        xi = eta = T( 1 ) / T( 3 );
        zeta     = T( 0 );
        return;
    }

    const T rho = mu( 0 ) + mu( 1 ) + mu( 2 );
    xi          = mu( 1 ) / sum;
    eta         = mu( 2 ) / sum;

    const T r1 = coords_radii( subdomain, cell.r );
    const T r2 = coords_radii( subdomain, cell.r + 1 );
    zeta       = Kokkos::clamp( T( 2 ) * ( rho - r1 ) / ( r2 - r1 ) - T( 1 ), T( -1 ), T( 1 ) );
}

/// @brief Locates a physical point in the wedge mesh of one subdomain, starting the search at `seed`.
///
/// Walks laterally across triangle edges towards the most negative barycentric coordinate, then locates the
/// radial cell. `max_lateral_steps` bounds the walk; it should be chosen as roughly `ceil(CFL) + 2`.
///
/// Points outside `[rho_clamp_min, rho_clamp_max]` are, if `clamp_radially` is set, pulled back onto that
/// interval (the physically correct closure for a no-penetration boundary) and `clamped_radially` is reported;
/// otherwise they are reported as not found. Pass the *physical* shell radii here -- for a ghosted radii array
/// these are not the array ends.
template < typename T, typename CoordsShellType, typename CoordsRadiiType, typename LateralValidityType >
KOKKOS_INLINE_FUNCTION LocateResult< T > locate_point(
    const dense::Vec< T, 3 >& X,
    const int                 subdomain,
    const WedgeCell&          seed,
    const CoordsShellType&    coords_shell,
    const CoordsRadiiType&    coords_radii,
    const IndexBounds&        bounds,
    const int                 max_lateral_steps,
    const T                   eps,
    const bool                clamp_radially,
    const T                   rho_clamp_min,
    const T                   rho_clamp_max,
    const LateralValidityType& lateral_valid )
{
    LocateResult< T > result;

    WedgeCell          cell = seed;
    dense::Vec< T, 3 > mu;
    T                  rho = T( 0 );

    // ---- lateral walk -------------------------------------------------------------------------------------
    // Visibility walk on the spherical triangulation: step across the edge opposite the most negative cone
    // coordinate until all three are non-negative.
    bool lateral_found = false;

    const T scale = X.norm();

    // The seed must be usable; callers are expected to seed with a wholly owned cell.
    if ( !wedge_lateral_nodes_valid( cell, subdomain, lateral_valid ) )
    {
        result.cell              = cell;
        result.escaped_laterally = true;
        result.hit_invalid_node  = true;
        return result;
    }

    for ( int step = 0; step <= max_lateral_steps; ++step )
    {
        wedge_lateral_cone_coords( X, subdomain, cell, coords_shell, mu );

        // Order the three candidate edge crossings by how far outside the triangle the point is.
        int order[3] = { 0, 1, 2 };
        for ( int a = 0; a < 2; ++a )
        {
            for ( int b = a + 1; b < 3; ++b )
            {
                if ( mu( order[b] ) < mu( order[a] ) )
                {
                    const int t = order[a];
                    order[a]    = order[b];
                    order[b]    = t;
                }
            }
        }

        if ( mu( order[0] ) >= -eps * scale )
        {
            rho           = mu( 0 ) + mu( 1 ) + mu( 2 );
            lateral_found = true;
            break;
        }

        // Prefer the edge we are furthest outside of, but fall back to the others rather than giving up: a
        // neighbour may be out of bounds or touch a degenerate ghost corner while another still leads to the
        // wedge that contains the point.
        bool moved = false;
        for ( int k = 0; k < 3 && !moved; ++k )
        {
            if ( mu( order[k] ) >= -eps * scale )
                break; // not outside this edge; crossing it would move away from the point

            const WedgeCell next = wedge_edge_neighbor( cell, order[k] );

            // The hex cell (x, y) spans nodes x..x+1 and y..y+1, so the last cell index is num_nodes - 2.
            if ( next.x < 0 || next.y < 0 || next.x > bounds.num_nodes_x - 2 || next.y > bounds.num_nodes_y - 2 )
                continue;

            if ( !wedge_lateral_nodes_valid( next, subdomain, lateral_valid ) )
            {
                result.hit_invalid_node = true;
                continue;
            }

            cell  = next;
            moved = true;
        }

        if ( !moved )
        {
            result.cell              = cell;
            result.escaped_laterally = true;
            return result;
        }
    }

    if ( !lateral_found )
    {
        result.cell              = cell;
        result.escaped_laterally = true;
        return result;
    }

    // ---- radial search ------------------------------------------------------------------------------------
    // coords_radii( subdomain, . ) is monotonically increasing.
    const int num_cells_r = bounds.num_nodes_r - 1;

    // The clamp bounds are the *physical* shell radii, which for a ghosted radii array differ from the array
    // ends: radial ghosts beyond the CMB / surface are extrapolated so that the array stays monotone, but a
    // departure point must never be interpolated there.
    const T r_inner = rho_clamp_min;
    const T r_outer = rho_clamp_max;

    if ( rho < r_inner )
    {
        if ( !clamp_radially )
        {
            result.cell = cell;
            return result;
        }
        rho                      = r_inner;
        result.clamped_radially = true;
    }
    else if ( rho > r_outer )
    {
        if ( !clamp_radially )
        {
            result.cell = cell;
            return result;
        }
        rho                      = r_outer;
        result.clamped_radially = true;
    }

    int lo = 0;
    int hi = num_cells_r - 1;
    while ( lo < hi )
    {
        const int mid = ( lo + hi + 1 ) / 2;
        if ( coords_radii( subdomain, mid ) <= rho )
            lo = mid;
        else
            hi = mid - 1;
    }
    cell.r = lo;

    const T r1 = coords_radii( subdomain, cell.r );
    const T r2 = coords_radii( subdomain, cell.r + 1 );

    const T inv_rho = T( 1 ) / ( mu( 0 ) + mu( 1 ) + mu( 2 ) );

    result.cell  = cell;
    result.xi    = mu( 1 ) * inv_rho;
    result.eta   = mu( 2 ) * inv_rho;
    result.zeta  = T( 2 ) * ( rho - r1 ) / ( r2 - r1 ) - T( 1 );
    result.found = true;

    return result;
}

/// @brief Evaluates a Q1 scalar wedge field at reference coordinates inside a wedge cell.
template < typename T, typename FieldViewType >
KOKKOS_INLINE_FUNCTION T evaluate_q1_scalar(
    const FieldViewType& field,
    const int            subdomain,
    const WedgeCell&     cell,
    const T              xi,
    const T              eta,
    const T              zeta )
{
    int nx[3], ny[3];
    wedge_lateral_node_indices( cell, nx, ny );

    T value = T( 0 );
    for ( int j = 0; j < num_nodes_per_wedge; ++j )
    {
        const int lateral = j % 3;
        const int radial  = j / 3;
        value += field( subdomain, nx[lateral], ny[lateral], cell.r + radial ) * shape_lat( j, xi, eta ) *
                 shape_rad( j, zeta );
    }
    return value;
}

/// @brief Evaluates a Q1 vector wedge field at reference coordinates inside a wedge cell.
template < typename T, int VecDim, typename FieldViewType >
KOKKOS_INLINE_FUNCTION dense::Vec< T, VecDim > evaluate_q1_vec(
    const FieldViewType& field,
    const int            subdomain,
    const WedgeCell&     cell,
    const T              xi,
    const T              eta,
    const T              zeta )
{
    int nx[3], ny[3];
    wedge_lateral_node_indices( cell, nx, ny );

    dense::Vec< T, VecDim > value;
    for ( int d = 0; d < VecDim; ++d )
        value( d ) = T( 0 );

    for ( int j = 0; j < num_nodes_per_wedge; ++j )
    {
        const int lateral = j % 3;
        const int radial  = j / 3;
        const T   weight  = shape_lat( j, xi, eta ) * shape_rad( j, zeta );

        for ( int d = 0; d < VecDim; ++d )
        {
            value( d ) += field( subdomain, nx[lateral], ny[lateral], cell.r + radial, d ) * weight;
        }
    }
    return value;
}

/// @brief Forward map of a wedge cell: reference coordinates -> physical point.
///
/// Provided here so that tests can round-trip against \ref locate_point without pulling in the kernel helpers.
template < typename T, typename CoordsShellType, typename CoordsRadiiType >
KOKKOS_INLINE_FUNCTION dense::Vec< T, 3 > wedge_forward_map(
    const int              subdomain,
    const WedgeCell&       cell,
    const CoordsShellType& coords_shell,
    const CoordsRadiiType& coords_radii,
    const T                xi,
    const T                eta,
    const T                zeta )
{
    int nx[3], ny[3];
    wedge_lateral_node_indices( cell, nx, ny );

    dense::Vec< T, 3 > p[3];
    for ( int v = 0; v < 3; ++v )
    {
        for ( int d = 0; d < 3; ++d )
        {
            p[v]( d ) = coords_shell( subdomain, nx[v], ny[v], d );
        }
    }

    return forward_map(
        p[0],
        p[1],
        p[2],
        coords_radii( subdomain, cell.r ),
        coords_radii( subdomain, cell.r + 1 ),
        xi,
        eta,
        zeta );
}

} // namespace terra::fe::wedge::sl
