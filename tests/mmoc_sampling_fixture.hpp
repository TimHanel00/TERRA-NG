#pragma once
#include "terra/fe/wedge/sl/ghosted_geometry.hpp"
#include "terra/grid/shell/mmoc_point_sampling.hpp"
#include "terra/communication/shell/communication_plan.hpp"
#include "terra/util/init.hpp"
#include <iostream>
#include <limits>
#include <memory>

using namespace terra;
namespace sl = terra::fe::wedge::sl;
using Vec3 = dense::Vec< double, 3 >;
using Domain = grid::shell::DistributedDomain;

struct Fixture
{
    grid::shell::SubdomainToRankDistributionFunction owner;
    Domain domain;
    grid::Grid4DDataScalar< grid::NodeOwnershipFlag > mask;
    grid::Grid3DDataVec< double, 3 > coords;
    grid::Grid2DDataScalar< double > radii;
    sl::GhostExchange ghosts;
    grid::Grid4DDataScalar< double > field, ghost_field;
    grid::Grid4DDataVec< double, 3 > u, old_u, ghost_u, ghost_old_u;
    std::unique_ptr< grid::shell::MMOCPointSampler< double > > sampler;
    std::unique_ptr< sl::TiledInterpolator< double > > interpolate;
    Kokkos::View< Vec3* > positions;
    Kokkos::View< double* > shared, reference;
    Kokkos::View< Vec3* > shared_velocity, reference_velocity;

    Fixture( MPI_Comm comm, grid::shell::SubdomainToRankDistributionFunction owner )
        : owner( owner ), domain( Domain::create_uniform_on_comm( comm, 3, grid::shell::uniform_shell_radii( 0.5, 1.0, 9 ), 1, 1, owner ) ),
          mask( grid::setup_node_ownership_mask_data( domain ) ),
          coords( grid::shell::subdomain_unit_sphere_single_shell_coords< double >( domain ) ),
          radii( grid::shell::subdomain_shell_radii< double >( domain ) ), ghosts( domain ),
          field( "field", mask.extent( 0 ), 5, 5, 5 ), ghost_field( ghosts.allocate_scalar< double >( "ghost_field" ) ),
          u( "u", mask.extent( 0 ), 5, 5, 5 ), old_u( "old_u", mask.extent( 0 ), 5, 5, 5 ),
          ghost_u( ghosts.allocate_vec< double, 3 >( "ghost_u" ) ),
          ghost_old_u( ghosts.allocate_vec< double, 3 >( "ghost_old_u" ) ),
          positions( "positions", mask.size() ), shared( "shared", mask.size() ), reference( "reference", mask.size() ),
          shared_velocity( "shared_velocity", mask.size() ), reference_velocity( "reference_velocity", mask.size() )
    { initialize(); }

    // CUDA extended lambdas require an addressable enclosing function, rather than a constructor.
    void initialize()
    {
        const auto c = coords; const auto r = radii; const auto f = field; const auto v = u; const auto o = old_u;
        Kokkos::parallel_for( "initialize", grid::shell::local_domain_md_range_policy_nodes( domain ),
            KOKKOS_LAMBDA( int s, int x, int y, int k ) {
                const auto p = grid::shell::coords( s, x, y, k, c, r );
                f( s, x, y, k ) = 1.0 + 0.1 * Kokkos::sin( 4 * p( 0 ) ) + 0.2 * p( 1 ) * p( 2 );
                v( s, x, y, k, 0 ) = -p( 1 ); v( s, x, y, k, 1 ) = p( 0 ); v( s, x, y, k, 2 ) = 0.1 * p( 2 );
                o( s, x, y, k, 0 ) = p( 1 ); o( s, x, y, k, 1 ) = 0.3 * p( 2 ); o( s, x, y, k, 2 ) = p( 0 );
            } );
        ghosts.fill( field, ghost_field );
        // Any accidental velocity ghost read must fail the finite-value checks, on either evaluator path.
        for ( int d = 0; d < 3; ++d )
        {
            Kokkos::deep_copy( ghost_u.comp_[d], std::numeric_limits< double >::quiet_NaN() );
            Kokkos::deep_copy( ghost_old_u.comp_[d], std::numeric_limits< double >::quiet_NaN() );
        }
        ghosts.copy_interior( u, ghost_u );
        ghosts.copy_interior( old_u, ghost_old_u );
        const auto ghost_coords = sl::ghosted_unit_sphere_coords< double >( domain, ghosts );
        const auto ghost_radii = sl::ghosted_shell_radii< double >( domain, ghosts );
        const auto valid = sl::ghosted_lateral_validity< double >( ghosts, ghost_coords );
        sampler = std::make_unique< grid::shell::MMOCPointSampler< double > >( domain, coords, radii, mask, owner );
        interpolate = std::make_unique< sl::TiledInterpolator< double > >( domain, ghost_field, ghost_u, ghost_old_u, ghost_radii, valid );
    }

    void sample( double angle )
    {
        const auto c = coords; const auto r = radii; const auto p = positions;
        Kokkos::parallel_for( "rotate", grid::shell::local_domain_md_range_policy_nodes( domain ),
            KOKKOS_LAMBDA( int s, int x, int y, int k ) {
                const auto a = grid::shell::coords( s, x, y, k, c, r );
                p( ( ( s * 5 + x ) * 5 + y ) * 5 + k ) = Vec3{
                    Kokkos::cos( angle ) * a( 0 ) - Kokkos::sin( angle ) * a( 1 ),
                    Kokkos::sin( angle ) * a( 0 ) + Kokkos::cos( angle ) * a( 1 ), a( 2 ) };
            } );
        sampler->sample( positions, reference, [&]( const auto& work, int n, const auto& out ) {
            interpolate->template evaluate< false >( work, n, out, 0.0, false );
        } );
        sampler->sample( positions, shared, [&]( const auto& work, int n, const auto& out ) {
            interpolate->template evaluate< false >( work, n, out, 0.0, true );
        } );
        sampler->sample( positions, reference_velocity, [&]( const auto& work, int n, const auto& out ) {
            interpolate->template evaluate< true >( work, n, out, 0.37, false );
        } );
        sampler->sample( positions, shared_velocity, [&]( const auto& work, int n, const auto& out ) {
            interpolate->template evaluate< true >( work, n, out, 0.37, true );
        } );
    }
};
