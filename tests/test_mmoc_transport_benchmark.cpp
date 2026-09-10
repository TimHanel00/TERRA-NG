#include "terra/fe/wedge/operators/shell/mmoc_transport.hpp"
#include "terra/util/init.hpp"
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <iostream>
#include <string>

using namespace terra;

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );
    Kokkos::Profiling::ScopedRegion application( "mmoc_application" );
    int level = 7, steps = 51, lateral = 1, radial = 1;
    bool shared = true;
    for ( int i = 1; i < argc; ++i )
    {
        const std::string option = argv[i];
        if ( option == "--global-interpolation" ) shared = false;
        else if ( option == "--no-output" ) {}
        else if ( option == "--level" && i + 1 < argc ) level = std::stoi( argv[++i] );
        else if ( option == "--steps" && i + 1 < argc ) steps = std::stoi( argv[++i] );
        else if ( option == "--lateral-subdivision" && i + 1 < argc ) lateral = std::stoi( argv[++i] );
        else if ( option == "--radial-subdivision" && i + 1 < argc ) radial = std::stoi( argv[++i] );
        else throw std::invalid_argument( "Unknown benchmark option: " + option );
    }
    if ( level < 1 || level > 13 || steps < 1 || lateral < 0 || radial < 0 || lateral > level || radial > level )
        throw std::invalid_argument( "Invalid MMOC benchmark configuration" );
    const auto domain = grid::shell::DistributedDomain::create_uniform( level, level, 0.5, 1.0, lateral, radial );
    const auto mask = grid::setup_node_ownership_mask_data( domain );
    const auto coords = grid::shell::subdomain_unit_sphere_single_shell_coords< double >( domain );
    const auto radii = grid::shell::subdomain_shell_radii< double >( domain );
    linalg::VectorQ1Scalar< double > temperature( "temperature", domain, mask );
    linalg::VectorQ1Vec< double, 3 > velocity( "velocity", domain, mask );
    linalg::VectorQ1Vec< double, 3 > previous( "previous_velocity", domain, mask );
    const auto t = temperature.grid_data();
    const auto u = velocity.grid_data(), old = previous.grid_data();
    Kokkos::parallel_for( "initialize_benchmark", grid::shell::local_domain_md_range_policy_nodes( domain ),
        KOKKOS_LAMBDA( int s, int x, int y, int r ) {
            const auto p = grid::shell::coords( s, x, y, r, coords, radii );
            const double distance = Kokkos::sqrt( p( 0 ) * p( 0 ) + ( p( 1 ) - 0.75 ) * ( p( 1 ) - 0.75 ) + p( 2 ) * p( 2 ) );
            t( s, x, y, r ) = Kokkos::max( 0.0, 1.0 - distance / 0.1 );
            u( s, x, y, r, 0 ) = old( s, x, y, r, 0 ) = -p( 1 );
            u( s, x, y, r, 1 ) = old( s, x, y, r, 1 ) = p( 0 );
            u( s, x, y, r, 2 ) = old( s, x, y, r, 2 ) = 0.0;
        } );
    fe::wedge::operators::shell::MMOCTransport< double > transport( domain, mask );
#ifdef TERRA_MMOC_HYBRID_BENCH
    transport.set_shared_interpolation( shared );
#else
    (void)shared;
#endif
    const double dt = 0.5 * grid::shell::min_radial_h( domain.domain_info().radii() );
    if ( mpi::rank( domain.comm() ) == 0 ) std::cout << "level=" << level << " steps=" << steps << " dt=" << dt
        << " lateral_subdivision=" << lateral << " radial_subdivision=" << radial << " substeps=1 RK4\n";
    Kokkos::fence();
    Kokkos::Profiling::pushRegion( "mmoc_timesteps" );
    std::uint64_t escapes = 0, queries = 0;
    for ( int step = 0; step < steps; ++step )
    {
        Kokkos::Timer timer;
        Kokkos::Profiling::pushRegion( "mmoc_sample" );
        transport.step( temperature, velocity, previous, dt, 1 );
        Kokkos::fence();
        Kokkos::Profiling::popRegion();
        double duration = timer.seconds();
        MPI_Allreduce( MPI_IN_PLACE, &duration, 1, MPI_DOUBLE, MPI_MAX, domain.comm() );
        escapes += transport.last_escapes();
#ifdef TERRA_MMOC_HYBRID_BENCH
        queries += transport.last_remote_queries();
#endif
        if ( mpi::rank( domain.comm() ) == 0 ) std::cout << "Maximum rank time for step: " << duration << '\n';
    }
    Kokkos::Profiling::popRegion();
    std::cout << "Rank " << mpi::rank( domain.comm() ) << " escaped_samples=" << escapes << " remote_queries=" << queries;
#ifdef TERRA_MMOC_HYBRID_BENCH
    std::cout << " sampling_device_buffer_bytes=" << transport.sampling_buffer_bytes();
#endif
    std::cout << std::endl;
    return 0;
}
