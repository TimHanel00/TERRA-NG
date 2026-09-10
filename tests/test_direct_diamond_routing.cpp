#include "terra/grid/shell/lateral_cell_lookup.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/util/init.hpp"

#include <iostream>
#include <cstdio>
#include <limits>

using namespace terra;
using Vec3 = dense::Vec< double, 3 >;

KOKKOS_INLINE_FUNCTION
bool radial_bounds_contain( const Vec3 ( &corners )[4], const Vec3& point, int level, double radius )
{
    // The known forward-map radius must remain in the sender's conservative interval
    // at every possible partition depth, including vertices and shared edges.
    constexpr double tolerance = 64 * std::numeric_limits< double >::epsilon();
    if ( point.norm() - tolerance > radius )
        return false;
    for ( int partition_level = 0; partition_level <= level; ++partition_level )
    {
        double coarse_radius = 0;
        const auto coarse = grid::shell::locate_lateral_cell( corners, point, partition_level, &coarse_radius );
        if ( !coarse.valid() || coarse_radius + tolerance < radius )
        {
            printf( "Radial bound failure: level=%d partition=%d valid=%d error=%.17g\n",
                level, partition_level, int( coarse.valid() ), coarse_radius - radius );
            return false;
        }
    }
    return true;
}

void test_level( int level )
{
    constexpr int samples = 64;
    const int n = 1 << level;
    grid::Grid3DDataVec< double, 3 > roots( "routing_roots", 10, 2, 2 );
    auto roots_host = Kokkos::create_mirror_view( roots );
    Kokkos::View< Vec3** > points( "routing_points", 10, samples );
    auto points_host = Kokkos::create_mirror_view( points );
    Kokkos::View< Vec3** > nodes( "routing_reference_nodes", 10, samples );
    auto nodes_host = Kokkos::create_mirror_view( nodes );
    Kokkos::View< Vec3*** > vertices( "routing_reference_vertices", 10, samples, 3 );
    auto vertices_host = Kokkos::create_mirror_view( vertices );
    Kokkos::View< int*** > cells( "routing_reference_cells", 10, samples, 3 );
    auto cells_host = Kokkos::create_mirror_view( cells );
    const int dx[2][3] = { { 0, 1, 0 }, { 1, 0, 1 } };
    const int dy[2][3] = { { 0, 0, 1 }, { 1, 1, 0 } };
    const double weights[3] = { 0.2, 0.3, 0.5 };

    for ( int diamond = 0; diamond < 10; ++diamond )
    {
        grid::shell::unit_sphere_single_shell_subdomain_coords< double >(
            roots_host, diamond, diamond, 2, 0, 1, 0, 1 );
        Vec3 corners[4];
        for ( int c = 0; c < 4; ++c )
            for ( int d = 0; d < 3; ++d )
                corners[c]( d ) = roots_host( diamond, c % 2, c / 2, d );
        const grid::shell::BaseCorners< double > base{ corners[0], corners[2], corners[1], corners[3], n };
        grid::shell::MemoizationCache< double > cache;
        for ( int sample = 0; sample < samples; ++sample )
        {
            const int x = sample < 8 ? ( ( sample / 2 ) % 2 ) * ( n - 1 ) : ( sample * 1031 + 17 ) % n;
            const int y = sample < 8 ? ( sample / 4 ) * ( n - 1 )
                : sample < 16 ? n - 1 - x : ( sample * 677 + 29 ) % n;
            const int wedge = sample % 2;
            cells_host( diamond, sample, 0 ) = x;
            cells_host( diamond, sample, 1 ) = y;
            cells_host( diamond, sample, 2 ) = wedge;
            Vec3 point{};
            for ( int v = 0; v < 3; ++v )
            {
                // Independent mesh-construction implementation; no direct-locator coordinates.
                const auto node = grid::shell::compute_node_recursive(
                    x + dx[wedge][v], y + dy[wedge][v], base, cache );
                vertices_host( diamond, sample, v ) = node;
                point = point + node * weights[v];
            }
            points_host( diamond, sample ) = point * 0.751;
            nodes_host( diamond, sample ) = grid::shell::compute_node_recursive( x, y, base, cache );
        }
    }
    Kokkos::deep_copy( roots, roots_host );
    Kokkos::deep_copy( points, points_host );
    Kokkos::deep_copy( nodes, nodes_host );
    Kokkos::deep_copy( vertices, vertices_host );
    Kokkos::deep_copy( cells, cells_host );
    int failures = 0;
    Kokkos::parallel_reduce(
        "check_direct_diamond_routing", 10 * samples,
        KOKKOS_LAMBDA( int i, int& errors ) {
            const int diamond = i / samples;
            const int sample = i % samples;
            grid::shell::detail::ReconstructedDiamondCoordinates< double > geometry;
            geometry.cells_per_side = n;
            for ( int c = 0; c < 4; ++c )
                for ( int d = 0; d < 3; ++d )
                    geometry.corners[c]( d ) = roots( diamond, c % 2, c / 2, d );
            const int x = cells( diamond, sample, 0 );
            const int y = cells( diamond, sample, 1 );
            if ( ( geometry.node_position( 0, x, y ) - nodes( diamond, sample ) ).norm() > 2e-14 )
                ++errors;
            double radius = 0;
            const auto cell = grid::shell::locate_diamond_cell_direct(
                geometry, n, points( diamond, sample ), 0.5, 1.0, radius );
            if ( !cell.valid() || cell.x != x || cell.y != y || cell.wedge != cells( diamond, sample, 2 ) ||
                 Kokkos::abs( radius - 0.751 ) > 1e-11 )
                ++errors;
            if ( !radial_bounds_contain( geometry.corners, points( diamond, sample ), level, 0.751 ) )
                ++errors;
            for ( int v = 0; v < 3; ++v )
            {
                // Vertices and edges include diamond boundaries and the diagonal between its two triangles.
                // Either adjacent cell is valid on a tie; verify reconstruction and containment instead of IDs.
                for ( int edge = 0; edge < 2; ++edge )
                {
                    const Vec3 point = ( edge ? ( vertices( diamond, sample, v ) +
                        vertices( diamond, sample, ( v + 1 ) % 3 ) ) * 0.5 : vertices( diamond, sample, v ) ) * 0.751;
                    if ( !radial_bounds_contain( geometry.corners, point, level, 0.751 ) )
                        ++errors;
                    const auto boundary_cell = grid::shell::locate_diamond_cell_direct(
                        geometry, n, point, 0.5, 1.0, radius );
                    if ( !boundary_cell.valid() || Kokkos::abs( radius - 0.751 ) > 1e-11 )
                    {
                        ++errors;
                        continue;
                    }
                    Vec3 mu;
                    const fe::wedge::sl::WedgeCell wedge_cell{
                        boundary_cell.x, boundary_cell.y, 0, boundary_cell.wedge };
                    fe::wedge::sl::wedge_lateral_cone_coords( point, 0, wedge_cell, geometry, mu );
                    if ( Kokkos::min( mu( 0 ), Kokkos::min( mu( 1 ), mu( 2 ) ) ) < -1e-11 )
                        ++errors;
                }
            }
        }, failures );
    if ( failures != 0 )
    {
        std::cerr << "Direct diamond routing failures: " << failures << " at level " << level << '\n';
        MPI_Abort( MPI_COMM_WORLD, 1 );
    }
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );
    for ( int level : { 0, 1, 2, 3, 4, 5, 6, 12, 13 } )
        test_level( level );
    if ( mpi::rank() == 0 )
        std::cout << "Direct diamond routing passed all diamonds at levels 0-6, 12, and 13.\n";
}
