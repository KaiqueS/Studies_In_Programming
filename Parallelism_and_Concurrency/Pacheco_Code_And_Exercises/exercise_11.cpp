#include <mpi.h>
#include <vector>
#include <iostream>
#include <random>

// 3.11 Finding prefix sums is a generalization of global sum. Rather than simply finding the sum of n values,
// x0 + x1 + · · · + xn−1, the prefix sums are the n partial sums x0 , x0 + x1 , x0 + x1 + x2 ,. . . , x0 + x1 + · · · + xn−1 .

// a. Devise a serial algorithm for computing the n prefix sums of an array with n elements.

std::vector<int> serial_prefix_sum( std::vector<int>& elements ){

    int partial{ 0 };
    std::vector<int> partial_sums{ };

    for( std::vector<int>::size_type i = 0; i < elements.size( ); ++i ){

        partial += elements[ i ];
        partial_sums.push_back( partial );
    }
     
     return partial_sums;
}

// b. Parallelize your serial algorithm for a system with n processes, each of which is storing one of the x_i’s.

std::vector<int> build_vector( int my_rank, int size ){

    std::vector<int> elements( size );

    if( my_rank == 0 ){

        std::random_device rand;
        std::uniform_int_distribution dist( 0, size );
        std::mt19937_64 rng( rand( ) );

        for( std::vector<int>::size_type i = 0; i < elements.size( ); ++i ){

            elements[ i ] = dist( rng );
        }
    }

    return elements;
}

void share_data( std::vector<int>& values, MPI_Comm comm ){

     MPI_Bcast( values.data( ), values.size( ), MPI_INT, 0, MPI_COMM_WORLD );
}

// I will perform a broadcast of the whole vector. But each process will only perform their respective partial sums
void parallel_prefix_sum( int my_rank, std::vector<int>& elements ){

    
}

// Represent the processes as nodes in a graph that is a tree. At the start, on the tree, each node has an edge
// from itself to the next node, except for the last node, which is linked to no other node. Assign to each process
// ( vector size / comm_sz ) elements, and let each node perform a serial prefix sum on their own elements. After
// this sum, send the GREATEST element from each node to the node its linked to. Sum this new element to its own
// elements. Now, for nodes i - 1, i, and i + 1, remove the edges ( i - 1, i ) and ( i, i + 1 ), and add the edge
// ( i - 1, i + 1 ), i.e., let node i - 1 be linked to its successor's successor. Repeat the sending of the greatest
// element. Repeat edge deletion and insertion until the LAST node becomes the successor of the FIRST node. Do one
// more sum. QED
void parallel_prefix_sum_enhanced( int my_rank, int comm_sz, std::vector<int>& elements, MPI_Comm comm ){

     int local_n = elements.size( ) / comm_sz;

    std::vector<int> local_partial( local_n );

    MPI_Scatter( elements.data( ), local_n, MPI_INT, local_partial.data( ), local_n, MPI_INT, 0, comm );

    local_partial = serial_prefix_sum( local_partial );

    for( std::vector<int>::size_type i = 0; i < comm_sz - 1; ++i ){

        int value{ 0 };

        MPI_Send( &local_partial[ local_partial.size( ) - 1 ], 1, MPI_INT, i + 1, 0, comm );
        MPI_Recv( &value, 1, MPI_INT, i, 0, comm, MPI_STATUS_IGNORE );


    }
}

// c. Suppose n = 2k for some positive integer k. Can you devise a serial algorithm and a parallelization of the serial algorithm
// so that the parallel algorithm requires only k communication phases? (You might want to look for this online.)

// d. MPI provides a collective communication function, MPI_Scan, that can be used to compute prefix sums:

// int MPI_Scan ( void∗ sendbuf_p, void∗ recvbuf_p, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm ) ;

// It operates on arrays with count elements; both sendbuf_p and recvbuf_p should refer to blocks of count elements of type datatype.
// The op argument is the same as op for MPI_Reduce. Write an MPI program that generates a random array of count elements on each MPI
// process, finds the prefix sums, and prints the results.

int main( ){

    int my_rank{ 0 }, comm_sz{ 0 };

    MPI_Init( nullptr, nullptr );
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &comm_sz );

    //std::vector<int> elem{ 1, 2, 3, 4, 5 };
    //std::vector<int> sums = serial_prefix_sum( elem );

    int vec_size{ 0 };

    if( my_rank == 0 ){ 

        printf( "Enter the size of the vector: " );
        scanf( "%d", &vec_size );
    }

    MPI_Bcast( &vec_size, 1, MPI_INT, 0, MPI_COMM_WORLD );

    std::vector<int> elements = build_vector( my_rank, vec_size );

    share_data( elements, MPI_COMM_WORLD );

    MPI_Finalize( );
}