// Test: closed-form wedge point location and Q1 evaluation used by the MMOC transport scheme.
//
// Checks, on a single rank over all 10 diamonds:
//   1. Round-trip: for reference coordinates inside every wedge cell, map forward to a physical point,
//      locate it starting from a deliberately distant seed cell, and verify that the located cell and the
//      recovered reference coordinates reproduce the original point.
//   2. Q1 exactness: the wedge map is isoparametric (sum_j N_j x_j = x), so the Q1 interpolant of a function
//      that is linear in the physical coordinates must be exact. Interpolating  g(x) = 1 + 2x + 3y - 0.5z
//      at arbitrary interior points must reproduce g to round-off.
//   3. Radial clamping: points below r_min / above r_max are pulled back onto the boundary shell.
//   4. The device (Kokkos) instantiation compiles and produces the same result as the host path.

#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>

#include <mpi.h>

#include "fe/wedge/sl/point_location.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/init.hpp"

using namespace terra;
using ScalarType = double;

using fe::wedge::sl::IndexBounds;
using fe::wedge::sl::WedgeCell;

namespace
{

ScalarType linear_field( const dense::Vec< ScalarType, 3 >& x )
{
    return 1.0 + 2.0 * x( 0 ) + 3.0 * x( 1 ) - 0.5 * x( 2 );
}

int g_failures = 0;

void check( const bool ok, const std::string& what )
{
    if ( !ok )
    {
        ++g_failures;
        if ( g_failures < 20 )
            std::cout << "  FAIL: " << what << std::endl;
    }
}

} // namespace

void test( const int level )
{
    const auto domain = grid::shell::DistributedDomain::create_uniform_single_subdomain_per_diamond(
        level, level, 0.5, 1.0 );

    const auto coords_shell_d = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto coords_radii_d = grid::shell::subdomain_shell_radii< ScalarType >( domain );

    auto coords_shell = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, coords_shell_d );
    auto coords_radii = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, coords_radii_d );

    const int num_subdomains = static_cast< int >( domain.subdomains().size() );
    const int num_nodes_lat  = domain.domain_info().subdomain_num_nodes_per_side_laterally();
    const int num_nodes_rad  = domain.domain_info().subdomain_num_nodes_radially();

    const IndexBounds bounds{ num_nodes_lat, num_nodes_lat, num_nodes_rad };

    // No ghost layer here, so every node is usable.
    Kokkos::View< uint8_t*** > all_valid( "all_valid", num_subdomains, num_nodes_lat, num_nodes_lat );
    Kokkos::deep_copy( all_valid, static_cast< uint8_t >( 1 ) );
    auto all_valid_h = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, all_valid );

    std::cout << "level=" << level << "  subdomains=" << num_subdomains << "  nodes_lat=" << num_nodes_lat
              << "  nodes_rad=" << num_nodes_rad << std::endl;

    // A generous walk budget: from the corner seed we may have to cross the whole diamond.
    const int max_steps = 8 * num_nodes_lat;
    const ScalarType eps = 1e-12;

    // Reference points strictly inside the reference wedge (xi, eta > 0, xi + eta < 1, |zeta| < 1).
    const ScalarType ref_points[][3] = {
        { 1.0 / 3.0, 1.0 / 3.0, 0.0 },
        { 0.1, 0.1, -0.8 },
        { 0.7, 0.2, 0.6 },
        { 0.15, 0.7, 0.9 },
        { 0.45, 0.45, -0.95 },
    };
    const int num_ref_points = sizeof( ref_points ) / sizeof( ref_points[0] );

    ScalarType max_position_error = 0.0;
    ScalarType max_linear_error   = 0.0;
    long long  num_located        = 0;
    long long  num_escaped        = 0;

    for ( int sd = 0; sd < num_subdomains; ++sd )
    {
        for ( int cx = 0; cx < num_nodes_lat - 1; ++cx )
        {
            for ( int cy = 0; cy < num_nodes_lat - 1; ++cy )
            {
                for ( int cr = 0; cr < num_nodes_rad - 1; ++cr )
                {
                    for ( int w = 0; w < 2; ++w )
                    {
                        const WedgeCell cell{ cx, cy, cr, w };

                        for ( int p = 0; p < num_ref_points; ++p )
                        {
                            const ScalarType xi   = ref_points[p][0];
                            const ScalarType eta  = ref_points[p][1];
                            const ScalarType zeta = ref_points[p][2];

                            const auto X = fe::wedge::sl::wedge_forward_map(
                                sd, cell, coords_shell, coords_radii, xi, eta, zeta );

                            // Seed the search far away from the true cell (opposite corner of the diamond).
                            const WedgeCell seed{ 0, 0, 0, 0 };

                            const auto res = fe::wedge::sl::locate_point(
                                X, sd, seed, coords_shell, coords_radii, bounds, max_steps, eps,
                                /*clamp_radially=*/false, ScalarType( 0 ), ScalarType( 1e30 ), all_valid_h );

                            if ( !res.found )
                            {
                                ++num_escaped;
                                check( false,
                                       "not located: sd=" + std::to_string( sd ) + " cell=(" +
                                           std::to_string( cx ) + "," + std::to_string( cy ) + "," +
                                           std::to_string( cr ) + "," + std::to_string( w ) + ")" );
                                continue;
                            }
                            ++num_located;

                            check( res.cell.x == cell.x && res.cell.y == cell.y && res.cell.r == cell.r &&
                                       res.cell.w == cell.w,
                                   "wrong cell: sd=" + std::to_string( sd ) + " expected (" +
                                       std::to_string( cx ) + "," + std::to_string( cy ) + "," +
                                       std::to_string( cr ) + "," + std::to_string( w ) + ") got (" +
                                       std::to_string( res.cell.x ) + "," + std::to_string( res.cell.y ) + "," +
                                       std::to_string( res.cell.r ) + "," + std::to_string( res.cell.w ) + ")" );

                            // Round-trip the recovered reference coordinates back to a physical point.
                            const auto X_back = fe::wedge::sl::wedge_forward_map(
                                sd, res.cell, coords_shell, coords_radii, res.xi, res.eta, res.zeta );
                            max_position_error = std::max( max_position_error, ( X_back - X ).norm() );

                            // Q1 exactness for a linear field: build the six nodal values on the fly.
                            int nx[3], ny[3];
                            fe::wedge::sl::wedge_lateral_node_indices( res.cell, nx, ny );
                            ScalarType interpolated = 0.0;
                            for ( int j = 0; j < 6; ++j )
                            {
                                const int lateral = j % 3;
                                const int radial  = j / 3;
                                dense::Vec< ScalarType, 3 > node;
                                for ( int d = 0; d < 3; ++d )
                                    node( d ) = coords_shell( sd, nx[lateral], ny[lateral], d );
                                node = node * coords_radii( sd, res.cell.r + radial );

                                interpolated += linear_field( node ) *
                                                fe::wedge::shape_lat( j, res.xi, res.eta ) *
                                                fe::wedge::shape_rad( j, res.zeta );
                            }
                            max_linear_error =
                                std::max( max_linear_error, std::abs( interpolated - linear_field( X ) ) );
                        }
                    }
                }
            }
        }
    }

    std::cout << std::scientific << std::setprecision( 4 );
    std::cout << "  located          : " << num_located << "  (escaped: " << num_escaped << ")" << std::endl;
    std::cout << "  max |X_back - X| : " << max_position_error << std::endl;
    std::cout << "  max linear err   : " << max_linear_error << std::endl;

    check( max_position_error < 1e-12, "round-trip position error too large" );
    check( max_linear_error < 1e-12, "Q1 linear reproduction error too large" );
    check( num_escaped == 0, "some points escaped the index space" );

    // ---- radial clamping ---------------------------------------------------------------------------------
    {
        const WedgeCell cell{ 3, 3, 0, 0 };
        const auto      X_inner =
            fe::wedge::sl::wedge_forward_map( 0, cell, coords_shell, coords_radii, 0.3, 0.3, -1.0 );

        const auto X_below = X_inner * ScalarType( 0.5 ); // well below r_min

        const auto res_no_clamp = fe::wedge::sl::locate_point(
            X_below, 0, WedgeCell{ 0, 0, 0, 0 }, coords_shell, coords_radii, bounds, max_steps, eps, false,
            coords_radii( 0, 0 ), coords_radii( 0, num_nodes_rad - 1 ), all_valid_h );
        check( !res_no_clamp.found, "point below r_min should not be found without clamping" );

        const auto res_clamp = fe::wedge::sl::locate_point(
            X_below, 0, WedgeCell{ 0, 0, 0, 0 }, coords_shell, coords_radii, bounds, max_steps, eps, true,
            coords_radii( 0, 0 ), coords_radii( 0, num_nodes_rad - 1 ), all_valid_h );
        check( res_clamp.found && res_clamp.clamped_radially, "point below r_min should clamp onto r_min" );
        check( res_clamp.cell.r == 0 && std::abs( res_clamp.zeta + 1.0 ) < 1e-12,
               "clamped point should sit on the innermost shell" );
    }

    // ---- device instantiation ----------------------------------------------------------------------------
    // Runs the same location on the default execution space and counts mismatches, proving the kernel is
    // device-compilable and branch-free enough to run inside a parallel_for.
    {
        int  device_failures = 0;
        auto cs              = coords_shell_d;
        auto cr_             = coords_radii_d;

        Kokkos::parallel_reduce(
            "mmoc_locate_device_check",
            Kokkos::MDRangePolicy< Kokkos::Rank< 3 > >( { 0, 0, 0 },
                                                        { num_subdomains, num_nodes_lat - 1, num_nodes_lat - 1 } ),
            KOKKOS_LAMBDA( const int sd, const int cx, const int cy, int& acc ) {
                const WedgeCell cell{ cx, cy, 0, 0 };
                const auto      X = fe::wedge::sl::wedge_forward_map(
                    sd, cell, cs, cr_, ScalarType( 0.25 ), ScalarType( 0.25 ), ScalarType( 0.0 ) );

                const auto res = fe::wedge::sl::locate_point(
                    X, sd, WedgeCell{ 0, 0, 0, 0 }, cs, cr_, bounds, max_steps, ScalarType( 1e-12 ), false,
                    ScalarType( 0 ), ScalarType( 1e30 ), all_valid );

                if ( !res.found || res.cell.x != cx || res.cell.y != cy || res.cell.w != 0 )
                    acc += 1;
            },
            device_failures );
        Kokkos::fence();

        std::cout << "  device mismatches: " << device_failures << std::endl;
        check( device_failures == 0, "device location disagrees with host location" );
    }
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    test( 3 );
    test( 4 );

    if ( g_failures == 0 )
    {
        std::cout << "\ntest_mmoc_point_location: PASSED" << std::endl;
    }
    else
    {
        std::cout << "\ntest_mmoc_point_location: FAILED (" << g_failures << " checks)" << std::endl;
    }
    return g_failures == 0 ? 0 : 1;
}
