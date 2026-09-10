#include "mmoc_sampling_fixture.hpp"

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );
    const int ranks = mpi::num_processes();
    const auto owner = [ranks]( const grid::shell::SubdomainInfo& id, int, int ) {
        return mpi::MPIRank( id.subdomain_r() % ranks );
    };
    Fixture test( MPI_COMM_WORLD, owner );
    Fixture oracle( MPI_COMM_SELF, []( const auto&, int, int ) { return mpi::MPIRank( 0 ); } );
    const grid::shell::SubdomainInfo source_id( 0, 0, 0, 0 ), target_id( 0, 0, 0, 1 );
    const auto c = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, oracle.coords );
    const auto r = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, oracle.radii );
    for ( bool halo_landing : { false, true } )
    {
        const auto id = halo_landing ? target_id : source_id;
        const int s_ref = std::get< 0 >( oracle.domain.subdomains().at( id ) );
        const sl::WedgeCell cell{ 1, 1, halo_landing ? 0 : 3, 0 };
        const Vec3 point = sl::wedge_forward_map( s_ref, cell, c, r, 0.25, 0.3, 0.0 );
        const auto coords = test.coords; const auto radii = test.radii; const auto positions = test.positions;
        Kokkos::parallel_for( "identity_queries", grid::shell::local_domain_md_range_policy_nodes( test.domain ),
            KOKKOS_LAMBDA( int s, int x, int y, int k ) {
                positions( ( ( s * 5 + x ) * 5 + y ) * 5 + k ) = grid::shell::coords( s, x, y, k, coords, radii );
            } );
        int origin = -1;
        if ( mpi::rank() == 0 )
        {
            const int source = std::get< 0 >( test.domain.subdomains().at( source_id ) );
            origin = ( ( source * 5 + 2 ) * 5 + 2 ) * 5 + 2;
            Kokkos::parallel_for( "target_query", 1, KOKKOS_LAMBDA( int ) { positions( origin ) = point; } );
        }
        test.sampler->sample( positions, test.shared, [&]( const auto& work, int n, const auto& output ) {
            test.interpolate->template evaluate< false >( work, n, output, 0.0, true );
        } );
        std::uint64_t remote = test.sampler->last_remote_queries();
        MPI_Allreduce( MPI_IN_PLACE, &remote, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD );
        const std::uint64_t expected_remote = halo_landing && ranks > 1 ? 1 : 0;
        if ( remote != expected_remote )
        {
            std::cerr << "halo_landing=" << halo_landing << " expected remote=" << expected_remote << " got=" << remote << '\n';
            MPI_Abort( MPI_COMM_WORLD, 1 );
        }
        // A nonlinear owner stencil must actually read its radial ghost row.
        if ( !halo_landing && mpi::rank() == 0 )
        {
            const int source = std::get< 0 >( test.domain.subdomains().at( source_id ) );
            const auto ghost_field = test.ghost_field;
            const auto before = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, test.shared );
            const double original = before( origin );
            Kokkos::parallel_for( "change_stencil_ghost", 49, KOKKOS_LAMBDA( int i ) {
                ghost_field( source, i / 7, i % 7, 6 ) += 0.02;
            } );
            // An already located owned-cell query avoids invoking any MPI operation here.
            Kokkos::View< sl::LocatedSample< double >* > work( "owned_boundary", 1 );
            Kokkos::View< double* > value( "owned_boundary_value", 1 );
            Kokkos::parallel_for( "owned_boundary_query", 1, KOKKOS_LAMBDA( int ) {
                work( 0 ) = { source, { 1, 1, 3, 0 }, 0.25, 0.3, 0.0 };
            } );
            test.interpolate->template evaluate< false >( work, 1, value, 0.0, true );
            auto changed = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, value );
            if ( !std::isfinite( changed( 0 ) ) || std::abs( changed( 0 ) - original ) < 1e-13 )
            {
                std::cerr << "Owned boundary interpolation did not use its ghost stencil\n";
                MPI_Abort( MPI_COMM_WORLD, 1 );
            }
        }
        // Restore immutable field values before the next test case.
        test.ghosts.fill( test.field, test.ghost_field );
    }
    if ( mpi::rank() == 0 ) std::cout << "Owned boundary uses ghost stencil; halo landing routes to owner.\n";
    return 0;
}
