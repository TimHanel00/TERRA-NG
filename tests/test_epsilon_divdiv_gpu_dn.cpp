#include <algorithm>
#include <cmath>
#include <iostream>
#include <span>

#include "fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/vector_q1.hpp"
#include "util/init.hpp"

namespace {

void compare_gpu_dn( int level, bool diagonal, terra::grid::shell::BoundaryConditions bcs, int mean )
{
    using namespace terra;
    using Op = fe::wedge::operators::shell::EpsilonDivDivKerngen< double >;
    const auto domain =
        grid::shell::DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, 0.5, 1.0 );
    const auto                    ownership = grid::setup_node_ownership_mask_data( domain );
    const auto                    boundary  = grid::shell::setup_boundary_mask_data( domain );
    const auto                    coords = grid::shell::subdomain_unit_sphere_single_shell_coords< double >( domain );
    const auto                    radii  = grid::shell::subdomain_shell_radii< double >( domain );
    linalg::VectorQ1Vec< double > src( "src", domain, ownership );
    linalg::VectorQ1Vec< double > reference_dst( "reference", domain, ownership );
    linalg::VectorQ1Vec< double > candidate_dst( "candidate", domain, ownership );
    linalg::VectorQ1Vec< double > error( "error", domain, ownership );
    linalg::VectorQ1Scalar< double > coefficient( "coefficient", domain, ownership );
    const auto                       src_data = src.grid_data();
    const auto                       k_data   = coefficient.grid_data();
    Kokkos::parallel_for(
        "initialize tile comparison",
        local_domain_md_range_policy_nodes( domain ),
        KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r ) {
            k_data( sd, x, y, r ) = 1.0 + 0.01 * x + 0.02 * y + 0.03 * r;
            for ( int d = 0; d < 3; ++d )
                src_data( sd, x, y, r, d ) = Kokkos::sin( 0.1 * ( x + 2 * y + 3 * r + d + sd ) );
        } );
    Kokkos::fence();

    using namespace fe::wedge::operators::shell;
    auto set_coefficient_mode = [&]( Op& op ) {
        if ( mean == 4 )
            op.set_q0_coefficient_field( coefficient.grid_data() );
        else if ( mean != 0 )
        {
            op.set_homogenize_eta_per_cell( true );
            op.set_eta_homogenization_mean( mean );
        }
    };
    Op reference( domain, coords, radii, boundary, coefficient.grid_data(), bcs, diagonal );
    reference.set_kernel_path( Op::KernelPath::Slow );
    set_coefficient_mode( reference );
    linalg::apply( reference, src, reference_dst );
    const double scale = std::max( 1.0, linalg::norm_inf( reference_dst ) );

    Op candidate( domain, coords, radii, boundary, coefficient.grid_data(), bcs, diagonal );
    set_coefficient_mode( candidate );
    linalg::apply( candidate, src, candidate_dst );
    linalg::lincomb( error, { 1.0, -1.0 }, { candidate_dst, reference_dst } );
    const double relative_error = linalg::norm_inf( error ) / scale;
    std::cout << "level=" << level << " diagonal=" << diagonal << " mean=" << mean
              << " relative_inf_error=" << relative_error << std::endl;
    if ( !std::isfinite( relative_error ) || relative_error > 1e-12 )
        Kokkos::abort( "EpsilonDivDiv optimization changed the operator result." );
}

} // namespace

int main( int argc, char** argv )
{
    terra::util::terra_initialize( &argc, &argv );
    using namespace terra::grid::shell;
    using enum BoundaryConditionFlag;
    using enum ShellBoundaryFlag;
    for ( const int mean : { 0, 1, 2, 3, 4 } )
        for ( const int level : { 1, 3, 4, 6, 7 } )
            for ( const bool diagonal : { false, true } )
                for ( const auto cmb_bc : { DIRICHLET, NEUMANN } )
                    for ( const auto surface_bc : { DIRICHLET, NEUMANN } )
                    {
                        BoundaryConditions bcs = { { CMB, cmb_bc }, { SURFACE, surface_bc } };
                        compare_gpu_dn( level, diagonal, bcs, mean );
                    }
    std::cout << "GPU_DN_COMPARISON_PASSED\n";
}
