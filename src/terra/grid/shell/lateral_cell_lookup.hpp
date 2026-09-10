#pragma once

#include "terra/grid/grid_types.hpp"
#include "terra/fe/wedge/sl/point_location.hpp"

#include <limits>

namespace terra::grid::shell {

/// Lateral cell (x,y) and triangle (wedge); combine with a radial interval for a 3D cell.
struct LateralCellLocation
{
    int x     = -1;
    int y     = -1;
    int wedge = -1;

    KOKKOS_INLINE_FUNCTION
    bool valid() const { return wedge >= 0; }
};

namespace detail {

// Generate one indexed node of TERRA's normalized-midpoint mesh without storing the remote mesh.
// Integer barycentric indices select its construction path; no physical containment search is performed.
// Working storage is constant. Coordinate generation, unlike the direct cell search, costs O(log N).
template < std::floating_point T >
struct ReconstructedDiamondCoordinates
{
    dense::Vec< T, 3 > corners[4];
    int cells_per_side;

    KOKKOS_INLINE_FUNCTION dense::Vec< T, 3 > node_position( int, int x, int y ) const
    {
        int length = cells_per_side;
        const bool lower = x + y <= length;
        auto p0 = corners[lower ? 0 : 3];
        auto p1 = corners[lower ? 1 : 2];
        auto p2 = corners[lower ? 2 : 1];
        int a = lower ? length - x - y : x + y - length;
        int b = lower ? x : length - x;
        int c = lower ? y : length - y;
        while ( true )
        {
            if ( a == length ) return p0;
            if ( b == length ) return p1;
            if ( c == length ) return p2;
            const int half = length / 2;
            if ( a >= half )
            {
                p1 = ( p0 + p1 ).normalized();
                p2 = ( p0 + p2 ).normalized();
                a -= half;
            }
            else if ( b >= half )
            {
                p0 = ( p0 + p1 ).normalized();
                p2 = ( p1 + p2 ).normalized();
                b -= half;
            }
            else if ( c >= half )
            {
                p0 = ( p0 + p2 ).normalized();
                p1 = ( p1 + p2 ).normalized();
                c -= half;
            }
            else
            {
                const auto m01 = ( p0 + p1 ).normalized();
                const auto m12 = ( p1 + p2 ).normalized();
                p2 = ( p2 + p0 ).normalized();
                p0 = m01;
                p1 = m12;
                const int next_a = half - c;
                const int next_b = half - a;
                c = half - b;
                a = next_a;
                b = next_b;
            }
            length = half;
        }
    }
};

// Only lateral cell indices and cone radius are needed during routing.
template < std::floating_point T >
struct RoutingRadii
{
    T inner;
    T outer;
    KOKKOS_INLINE_FUNCTION T operator()( int, int r ) const { return r == 0 ? inner : outer; }
};

struct ValidLateralNodes
{
    KOKKOS_INLINE_FUNCTION bool operator()( int, int, int ) const { return true; }
};

template < std::floating_point T >
struct LookupTriangle
{
    dense::Vec< int, 2 > index[3];
    dense::Vec< T, 3 > vertex[3];
};

// Positive margin means inside all three spherical edges; normalization makes tolerance independent of level.
template < std::floating_point T >
KOKKOS_INLINE_FUNCTION T triangle_margin( const LookupTriangle< T >& triangle, const dense::Vec< T, 3 >& point )
{
    T margin = T( 2 );
    for ( int i = 0; i < 3; ++i )
    {
        // Edge differences avoid cancellation between nearly parallel unit vectors.
        // Evaluate signed distances relative to the edge's vertex for the same reason.
        const auto normal = triangle.vertex[i].cross( triangle.vertex[( i + 1 ) % 3] - triangle.vertex[i] );
        const T length = normal.norm();
        const T reference = normal.dot( triangle.vertex[( i + 2 ) % 3] - triangle.vertex[i] );
        if ( !( length > T( 0 ) ) || reference == T( 0 ) )
            return T( -2 );
        const T distance = ( reference > T( 0 ) ? T( 1 ) : T( -1 ) ) * normal.dot( point - triangle.vertex[i] ) / length;
        margin = Kokkos::min( margin, distance );
    }
    return margin;
}

template < std::floating_point T, int Count >
KOKKOS_INLINE_FUNCTION int select_triangle(
    const LookupTriangle< T > ( &triangles )[Count], const dense::Vec< T, 3 >& point )
{
    constexpr T tolerance = T( 64 ) * std::numeric_limits< T >::epsilon();
    int selected = -1;
    T best_margin = -tolerance;
    for ( int i = 0; i < Count; ++i )
    {
        const T margin = triangle_margin( triangles[i], point );
        // Prefer the most interior candidate; exact ties retain the first one.
        if ( margin >= -tolerance && ( selected < 0 || margin > best_margin ) )
        {
            selected = i;
            best_margin = margin;
        }
    }
    return selected;
}

} // namespace detail

/// Find the triangle containing an arbitrary position by following TERRA's midpoint refinement.
/// Each level selects one of four children; no nearest-node or neighborhood search is needed.
/// refinement_level is relative to this patch: global lateral level minus its subdivision level.
/// Corners: (0,0), (1,0), (0,1), (1,1); wedge numbering matches wedge_surface_physical_coords.
/// Optionally return the physical wedge's cone radius; the caller locates the radial interval separately.
/// Boundary ties are deterministic within a patch; the caller handles subdomain ownership.
/// Failed containment returns invalid IDs. This requires TERRA's midpoint-refined geometry.
template < std::floating_point T >
KOKKOS_INLINE_FUNCTION LateralCellLocation locate_lateral_cell(
    const dense::Vec< T, 3 > ( &corners )[4], dense::Vec< T, 3 > point, const int refinement_level,
    T* cone_radius = nullptr )
{
    if ( refinement_level < 0 || refinement_level >= 31 )
        return {};
    const T radius = point.norm();
    if ( !( radius > T( 0 ) ) || !Kokkos::isfinite( radius ) )
        return {};
    // Use a direction for angular containment; the caller retains the physical position and radius.
    const auto direction = point * ( T( 1 ) / radius );

    const int n = 1 << refinement_level;
    using Index = dense::Vec< int, 2 >;
    using Triangle = detail::LookupTriangle< T >;
    // Start with the two triangles spanning the patch.
    const Triangle roots[2] = {
        { { Index{ 0, 0 }, Index{ n, 0 }, Index{ 0, n } }, { corners[0], corners[1], corners[2] } },
        { { Index{ n, n }, Index{ 0, n }, Index{ n, 0 } }, { corners[3], corners[2], corners[1] } }
    };
    const int root = detail::select_triangle( roots, direction );
    if ( root < 0 )
        return {};
    Triangle triangle = roots[root];

    // Follow only the containing child, one mesh refinement level at a time.
    for ( int level = 0; level < refinement_level; ++level )
    {
        Index midpoint_index[3];
        dense::Vec< T, 3 > midpoint[3];
        for ( int i = 0; i < 3; ++i )
        {
            for ( int d = 0; d < 2; ++d )
                midpoint_index[i]( d ) = ( triangle.index[i]( d ) + triangle.index[( i + 1 ) % 3]( d ) ) / 2;
            midpoint[i] = ( triangle.vertex[i] + triangle.vertex[( i + 1 ) % 3] ).normalized();
        }
        const Triangle children[4] = {
            { { triangle.index[0], midpoint_index[0], midpoint_index[2] },
              { triangle.vertex[0], midpoint[0], midpoint[2] } },
            { { midpoint_index[0], triangle.index[1], midpoint_index[1] },
              { midpoint[0], triangle.vertex[1], midpoint[1] } },
            { { midpoint_index[2], midpoint_index[1], triangle.index[2] },
              { midpoint[2], midpoint[1], triangle.vertex[2] } },
            { { midpoint_index[0], midpoint_index[1], midpoint_index[2] },
              { midpoint[0], midpoint[1], midpoint[2] } }
        };
        const int child = detail::select_triangle( children, direction );
        if ( child < 0 )
            return {};
        triangle = children[child];
    }

    if ( cone_radius )
    {
        // The plane ratio avoids inverting nearly parallel unit vectors at high partition levels.
        const auto normal = ( triangle.vertex[1] - triangle.vertex[0] ).cross(
            triangle.vertex[2] - triangle.vertex[0] );
        *cone_radius = point.dot( normal ) / triangle.vertex[0].dot( normal );
    }

    // Convert the final triangle's vertex indices to the field's cell and wedge indices.
    int x = triangle.index[0]( 0 );
    int y = triangle.index[0]( 1 );
    for ( int i = 1; i < 3; ++i )
    {
        x = Kokkos::min( x, triangle.index[i]( 0 ) );
        y = Kokkos::min( y, triangle.index[i]( 1 ) );
    }
    int wedge = 1;
    for ( int i = 0; i < 3; ++i )
        if ( triangle.index[i]( 0 ) == x && triangle.index[i]( 1 ) == y )
            wedge = 0;
    return { x, y, wedge };
}

/// Direct global-diamond lookup with stored or locally reconstructed node positions.
/// The caller has already checked diamond containment. No subdomain/refinement-tree search is needed.
template < std::floating_point T, typename Coordinates >
KOKKOS_INLINE_FUNCTION LateralCellLocation locate_diamond_cell_direct(
    const Coordinates& coords, int cells_per_side, const dense::Vec< T, 3 >& point,
    T r_min, T r_max, T& cone_radius )
{
    namespace sl = fe::wedge::sl;
    const sl::IndexBounds bounds{ cells_per_side + 1, cells_per_side + 1, 2 };
    const auto found = sl::locate_point_direct(
        point, 0, coords, detail::RoutingRadii< T >{ r_min, r_max },
        sl::corner_box_from_bounds( bounds ), bounds, 2, 4, T( 1e-12 ), true,
        r_min, r_max, detail::ValidLateralNodes{} );
    if ( !found.found )
        return {};
    dense::Vec< T, 3 > mu;
    sl::wedge_lateral_cone_coords( point, 0, found.cell, coords, mu );
    cone_radius = mu( 0 ) + mu( 1 ) + mu( 2 );
    return { found.cell.x, found.cell.y, found.cell.w };
}

} // namespace terra::grid::shell
