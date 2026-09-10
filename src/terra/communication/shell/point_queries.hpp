#pragma once
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <map>
#include <type_traits>
#include <vector>

namespace terra::communication::shell {

namespace detail {

inline void point_query_require( bool condition, MPI_Comm comm, const char* message )
{
    if ( !condition )
    {
        std::cerr << "Distributed point lookup: " << message << '\n';
        MPI_Abort( comm, 1 );
        std::abort();
    }
}

inline int point_query_count( size_t count, MPI_Comm comm )
{
    point_query_require( count <= INT_MAX, comm, "MPI count exceeds INT_MAX" );
    return static_cast< int >( count );
}

template < typename Record >
struct PointQueryDatatype
{
    MPI_Datatype type;
    explicit PointQueryDatatype()
    {
        static_assert( std::is_trivially_copyable_v< Record > );
        MPI_Type_contiguous( sizeof( Record ), MPI_BYTE, &type );
        MPI_Type_commit( &type );
    }
    ~PointQueryDatatype() { MPI_Type_free( &type ); }
    PointQueryDatatype( const PointQueryDatatype& ) = delete;
    PointQueryDatatype& operator=( const PointQueryDatatype& ) = delete;
};

} // namespace detail

/// Send queries to field owners, evaluate there, and return replies to the requesting ranks.
/// evaluate must return one reply per received query, in the same order.
/// Same-rank queries stay local; remote payloads use host buffers and a sparse MPI graph.
/// All ranks in comm participate, including empty ranks; the graph is rebuilt each call.
/// Reply tags use a private communicator. Records must be trivially copyable with the same ABI.
template < typename Query, typename Reply, typename Evaluate >
std::vector< Reply > exchange_point_queries(
    MPI_Comm comm, const std::map< int, std::vector< Query > >& outgoing, Evaluate evaluate )
{
    int rank = 0;
    MPI_Comm_rank( comm, &rank );
    // Separate local work from requests requiring MPI transport.
    std::vector< Query > local;
    std::vector< int > destinations;
    for ( const auto& [peer, queries] : outgoing )
    {
        if ( peer == rank )
            local = queries;
        else if ( !queries.empty() )
            destinations.push_back( peer );
    }

    // Discover incoming peers from the outgoing destinations supplied by all ranks.
    const int degree = detail::point_query_count( destinations.size(), comm );
    // Some MPI implementations validate array pointers even at degree zero.
    int empty = 0;
    MPI_Comm graph = MPI_COMM_NULL;
    Kokkos::Profiling::pushRegion( "mmoc_graph_create" );
    const int graph_status = MPI_Dist_graph_create(
        comm, 1, &rank, &degree, destinations.empty() ? &empty : destinations.data(),
        MPI_UNWEIGHTED, MPI_INFO_NULL, 0, &graph );
    Kokkos::Profiling::popRegion();
    detail::point_query_require( graph_status == MPI_SUCCESS, comm, "MPI graph creation failed" );
    MPI_Comm_set_errhandler( graph, MPI_ERRORS_ARE_FATAL );
    int incoming_degree = 0, outgoing_degree = 0, weighted = 0;
    MPI_Dist_graph_neighbors_count( graph, &incoming_degree, &outgoing_degree, &weighted );
    std::vector< int > sources( incoming_degree );
    destinations.resize( outgoing_degree );
    MPI_Dist_graph_neighbors(
        graph, incoming_degree, sources.empty() ? &empty : sources.data(), MPI_UNWEIGHTED,
        outgoing_degree, destinations.empty() ? &empty : destinations.data(), MPI_UNWEIGHTED );

    // Pack requests in graph-neighbor order and exchange counts before payloads.
    std::vector< int > send_counts( outgoing_degree ), receive_counts( incoming_degree );
    std::vector< int > send_offsets( outgoing_degree ), receive_offsets( incoming_degree );
    std::vector< Query > send_queries;
    for ( int i = 0; i < outgoing_degree; ++i )
    {
        const auto& batch = outgoing.at( destinations[i] );
        send_offsets[i] = detail::point_query_count( send_queries.size(), comm );
        send_counts[i] = detail::point_query_count( batch.size(), comm );
        send_queries.insert( send_queries.end(), batch.begin(), batch.end() );
    }
    detail::point_query_count( send_queries.size(), comm );
    Kokkos::Profiling::pushRegion( "mmoc_exchange_counts" );
    MPI_Neighbor_alltoall(
        send_counts.empty() ? &empty : send_counts.data(), 1, MPI_INT,
        receive_counts.empty() ? &empty : receive_counts.data(), 1, MPI_INT, graph );
    Kokkos::Profiling::popRegion();
    size_t received_count = 0;
    for ( int i = 0; i < incoming_degree; ++i )
    {
        receive_offsets[i] = detail::point_query_count( received_count, comm );
        received_count += receive_counts[i];
    }
    detail::point_query_count( received_count + local.size(), comm );
    std::vector< Query > received( received_count );
    detail::PointQueryDatatype< Query > query_type;
    detail::PointQueryDatatype< Reply > reply_type;
    Query empty_query{};
    Kokkos::Profiling::pushRegion( "mmoc_exchange_positions" );
    MPI_Neighbor_alltoallv(
        send_queries.empty() ? &empty_query : send_queries.data(),
        send_counts.empty() ? &empty : send_counts.data(), send_offsets.empty() ? &empty : send_offsets.data(), query_type.type,
        received.empty() ? &empty_query : received.data(),
        receive_counts.empty() ? &empty : receive_counts.data(), receive_offsets.empty() ? &empty : receive_offsets.data(), query_type.type, graph );

    Kokkos::Profiling::popRegion();
    // Evaluate local and received queries together; preserve their order for the return path.
    const size_t local_count = local.size();
    local.insert( local.end(), received.begin(), received.end() );
    auto evaluated = evaluate( local );
    detail::point_query_require( evaluated.size() == local.size(), comm, "missing query replies" );
    std::vector< Reply > result( local_count + send_queries.size() );
    std::copy_n( evaluated.begin(), local_count, result.begin() );
    // Reverse the request routes: receive our answers and send back answers computed here.
    Kokkos::Profiling::pushRegion( "mmoc_exchange_replies" );
    std::vector< MPI_Request > requests;
    requests.reserve( incoming_degree + outgoing_degree );
    for ( int i = 0; i < outgoing_degree; ++i )
    {
        MPI_Request request;
        MPI_Irecv(
            result.data() + local_count + send_offsets[i], send_counts[i], reply_type.type,
            destinations[i], 0, graph, &request );
        requests.push_back( request );
    }
    for ( int i = 0; i < incoming_degree; ++i )
    {
        MPI_Request request;
        MPI_Isend(
            evaluated.data() + local_count + receive_offsets[i], receive_counts[i], reply_type.type,
            sources[i], 0, graph, &request );
        requests.push_back( request );
    }
    MPI_Waitall( detail::point_query_count( requests.size(), comm ), requests.data(), MPI_STATUSES_IGNORE );
    Kokkos::Profiling::popRegion();
    MPI_Comm_free( &graph );
    return result;
}

} // namespace terra::communication::shell
