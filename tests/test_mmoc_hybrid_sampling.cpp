#include "mmoc_sampling_fixture.hpp"

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );
    const int ranks = mpi::num_processes();
    const bool mixed = argc > 1 && std::string( argv[1] ) == "--mixed";
    const bool all_root = argc > 1 && std::string( argv[1] ) == "--all-root";
    grid::shell::SubdomainToRankDistributionFunction owner = grid::shell::subdomain_to_rank_iterate_diamond_subdomains;
    if ( all_root ) owner = []( const auto&, int, int ) { return mpi::MPIRank( 0 ); };
    if ( mixed ) owner = [ranks]( const grid::shell::SubdomainInfo& id, int, int ) {
        return mpi::MPIRank( ranks == 1 || id.diamond_id() < 5 ? 0 : 1 + id.subdomain_r() % ( ranks - 1 ) );
    };
    Fixture test( MPI_COMM_WORLD, owner );
    Fixture oracle( MPI_COMM_SELF, []( const auto&, int, int ) { return mpi::MPIRank( 0 ); } );
    std::uint64_t remote_queries = 0;
    for ( double angle : { 0.0, 0.03, 0.19, 0.8 } )
    {
        test.sample( angle ); oracle.sample( angle );
        remote_queries += test.sampler->last_remote_queries();
        auto a = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, test.shared );
        auto b = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, test.reference );
        auto av = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, test.shared_velocity );
        auto bv = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, test.reference_velocity );
        auto mask = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, test.mask );
        grid::Grid4DDataScalar< double > full( "full", oracle.mask.extent( 0 ), 5, 5, 5 );
        const auto values = oracle.shared;
        Kokkos::parallel_for( "oracle_scatter", int( values.extent( 0 ) ), KOKKOS_LAMBDA( int i ) {
            full( i / 125, ( i / 25 ) % 5, ( i / 5 ) % 5, i % 5 ) = values( i );
        } );
        communication::shell::ShellBoundaryCommPlan< decltype( full ) > plan( oracle.domain, true, true );
        communication::shell::SubdomainNeighborhoodSendRecvBuffer< double > receive( oracle.domain );
        plan.exchange_and_reduce( full, receive );
        auto expected = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, full );
        double error = 0;
        for ( const auto& [id, data] : test.domain.subdomains() )
        {
            const int s = std::get< 0 >( data ), reference_s = std::get< 0 >( oracle.domain.subdomains().at( id ) );
            for ( int x = 0; x < 5; ++x ) for ( int y = 0; y < 5; ++y ) for ( int r = 0; r < 5; ++r )
            {
                if ( !util::has_flag( mask( s, x, y, r ), grid::NodeOwnershipFlag::OWNED ) ) continue;
                const int i = ( ( s * 5 + x ) * 5 + y ) * 5 + r;
                if ( !std::isfinite( a( i ) ) || !std::isfinite( b( i ) ) || !std::isfinite( expected( reference_s, x, y, r ) ) )
                    MPI_Abort( MPI_COMM_WORLD, 1 );
                for ( int d = 0; d < 3; ++d ) if ( !std::isfinite( av( i )( d ) ) || !std::isfinite( bv( i )( d ) ) )
                    MPI_Abort( MPI_COMM_WORLD, 1 );
                error = std::max( error, std::abs( a( i ) - b( i ) ) );
                error = std::max( error, std::abs( a( i ) - expected( reference_s, x, y, r ) ) );
                for ( int d = 0; d < 3; ++d ) error = std::max( error, std::abs( av( i )( d ) - bv( i )( d ) ) );
            }
        }
        MPI_Allreduce( MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
        if ( mpi::rank() == 0 ) std::cout << "angle=" << angle << " shared/owner error=" << error << '\n';
        if ( error > 1e-11 ) MPI_Abort( MPI_COMM_WORLD, 1 );
    }
    MPI_Allreduce( MPI_IN_PLACE, &remote_queries, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD );
    if ( ranks > 1 && !all_root && remote_queries == 0 ) MPI_Abort( MPI_COMM_WORLD, 1 );
    if ( all_root && remote_queries != 0 ) MPI_Abort( MPI_COMM_WORLD, 1 );
    if ( mpi::rank() == 0 ) std::cout << "Hybrid owner routing/shared interpolation passed; remote queries=" << remote_queries << '\n';
    return 0;
}
