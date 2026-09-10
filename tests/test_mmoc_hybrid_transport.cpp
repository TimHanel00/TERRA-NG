#include "mmoc_sampling_fixture.hpp"
#include "terra/fe/wedge/operators/shell/mmoc_transport.hpp"

struct Result
{
    grid::Grid4DDataScalar< double > field;
    std::uint64_t remote_queries;
};

Result advance( Fixture& fixture, bool shared )
{
    linalg::VectorQ1Scalar< double > temperature( "temperature", fixture.domain, fixture.mask );
    linalg::VectorQ1Vec< double, 3 > velocity( "velocity", fixture.domain, fixture.mask );
    linalg::VectorQ1Vec< double, 3 > old_velocity( "old_velocity", fixture.domain, fixture.mask );
    Kokkos::deep_copy( temperature.grid_data(), fixture.field );
    for ( int d = 0; d < 3; ++d )
    {
        Kokkos::deep_copy( velocity.grid_data().comp_[d], fixture.u.comp_[d] );
        Kokkos::deep_copy( old_velocity.grid_data().comp_[d], fixture.old_u.comp_[d] );
    }
    fe::wedge::operators::shell::MMOCTransport< double > transport(
        fixture.domain, fixture.mask, fe::wedge::operators::shell::TimeSteppingScheme::RK4, fixture.owner );
    transport.set_shared_interpolation( shared );
    std::uint64_t queries = 0;
    for ( int step = 0; step < 2; ++step )
    {
        transport.step( temperature, velocity, old_velocity, 0.25, 2 );
        queries += transport.last_remote_queries();
        if ( transport.last_escapes() != 0 ) MPI_Abort( fixture.domain.comm(), 1 );
    }
    return { temperature.grid_data(), queries };
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );
    const int ranks = mpi::num_processes();
    const auto owner = [ranks]( const grid::shell::SubdomainInfo& id, int, int ) {
        return mpi::MPIRank( ranks == 1 || id.diamond_id() < 5 ? 0 : 1 + id.subdomain_r() % ( ranks - 1 ) );
    };
    Fixture test( MPI_COMM_WORLD, owner );
    Fixture oracle( MPI_COMM_SELF, []( const auto&, int, int ) { return mpi::MPIRank( 0 ); } );
    const auto result = advance( test, true );
    const auto global = advance( test, false );
    const auto reference = advance( oracle, false );
    const auto a = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, result.field );
    const auto b = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, global.field );
    const auto c = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, reference.field );
    double error = 0;
    for ( const auto& [id, data] : test.domain.subdomains() )
    {
        const int s = std::get< 0 >( data ), reference_s = std::get< 0 >( oracle.domain.subdomains().at( id ) );
        for ( int x = 0; x < 5; ++x ) for ( int y = 0; y < 5; ++y ) for ( int r = 0; r < 5; ++r )
        {
            if ( !std::isfinite( a( s, x, y, r ) ) || !std::isfinite( b( s, x, y, r ) ) ||
                 !std::isfinite( c( reference_s, x, y, r ) ) ) MPI_Abort( MPI_COMM_WORLD, 1 );
            error = std::max( error, std::abs( a( s, x, y, r ) - b( s, x, y, r ) ) );
            error = std::max( error, std::abs( a( s, x, y, r ) - c( reference_s, x, y, r ) ) );
        }
    }
    auto queries = result.remote_queries;
    MPI_Allreduce( MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
    MPI_Allreduce( MPI_IN_PLACE, &queries, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD );
    if ( mpi::rank() == 0 ) std::cout << "RK4 two substeps, two timesteps: error=" << error << " remote queries=" << queries << '\n';
    if ( error > 1e-11 || ( ranks > 1 && queries == 0 ) ) MPI_Abort( MPI_COMM_WORLD, 1 );
    return 0;
}
