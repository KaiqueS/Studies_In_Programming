#include <mpi.h>
#include <iostream>
#include <vector>

/* 3.12 An alternative to a butterfly-structured allreduce is a “ring-pass” structure. In a ring-pass, if there
are p processes, each process q sends data to process q + 1, except that process p − 1 sends data to process
0. This is repeated until each process has the desired result. Thus we can implement allreduce with the following code:

    sum = temp_val = my_val;

    for( i = 1 ; i < p ; i ++ ){
        
        MPI_Sendrecv_replace( &temp_val, 1, MPI_INT, dest, sendtag, source, recvtag, comm, &status );
        
        sum += temp_val;
    } */

//  a.  Write an MPI program that implements this algorithm for allreduce. How does its performance compare to the
//      butterfly-structured allreduce?

void MPI_Ring_Allreduce( int my_rank, int comm_size, int my_val, int& result, MPI_Comm comm ){

    int sum{ my_val }, temp_val{ my_val };

    int dest = ( my_rank + 1 ) % comm_size;
    int source = ( my_rank + comm_size - 1 ) % comm_size;

    for( auto i = 1; i < comm_size; ++i ){

        /// ATTENTION: "source", the parameter, refers to the source from which the CALLING PROCESS
        ///            receives its message, and not to the calling process itself. I.e., do not read
        ///            "source" as "I, the source of the message I'm sending", but instead as "the source
        ///            of the message I'm receiving".
        MPI_Sendrecv_replace( &temp_val, 1, MPI_INT, dest, 0, source, 0, comm, MPI_STATUS_IGNORE );

        sum += temp_val;
    }

    result = sum;
}

// b.  Modify the MPI program you wrote in the first part so that it implements prefix sums.

/*  SKETCH: with a ring-pass structure with k-elements, every element gets added to every other element
///         after k-1 steps. But, at every step, one element has a correct partial sum. E.g.: at step 0,
///         the element 0 holds the correct partial sum. At step 1, element 1 holds the sum k_0 + k_1 = x_2,
///         because it receives the value from element 0 and adds it to its own element. On the next step,
///         element 2 adds the values from element 0 and 1 = k_0 + k_1 + k_2 = x_3. And so on, so, at every
///         step i, we get the current sum from process i and push it into our array for partial sums. At
///         step k-1, this array has all partial sums. QED */

/// ANSWER: the above sketch is correct, but the solution below is simpler - just let process 0 receive
///         from its succeeding processes, instead of sending to them. Add this received value to its
///         own value, and insert it into the prefixes array. This way, at each iteration process' 0
///         sum represents all the correct partial sums in order. QED
std::vector<int> prefix_sum( int my_rank, int comm_size, int my_val, MPI_Comm comm ){

    int sum{ my_val }, temp_val{ my_val };

    int dest = ( my_rank + 1 ) % comm_size;
    int source = ( my_rank + comm_size - 1 ) % comm_size;

    std::vector<int> prefixes{ };
    prefixes.push_back( my_val );

    // JUST REVERT THE RING PASS, THIS WAY, PROCESS 0 WILL GET ALL PARTIAL SUMS IN ORDER!
    for( auto i = 1; i < comm_size; ++i ){

        // Send to my predecessor. Receive from successor. I.e., invert dest-source on MPI_Sendrecv_replace
        MPI_Sendrecv_replace( &temp_val, 1, MPI_INT, source, 0, dest, 0, comm, MPI_STATUS_IGNORE );

        sum += temp_val;

        prefixes.push_back( sum );
    }

    return prefixes;
}

// For this to work, elements.size = k * comm_size, i.e., elements.size must be divisible by comm_size
std::vector<int> generalized_prefix_sum( std::vector<int>& elements, int my_rank, int comm_size, MPI_Comm comm ){

    int sum{ elements[ my_rank ] }, temp_val{ elements[ my_rank ] };

    int dest = ( my_rank + 1 ) % comm_size;
    int source = ( my_rank + comm_size - 1 ) % comm_size;

    std::vector<int> prefixes{ };
    prefixes.push_back( elements[ my_rank ] );

    for( auto i = 1; i < elements.size( ); ++i ){

        MPI_Sendrecv_replace( &temp_val, 1, MPI_INT, source, 0, dest, 0, comm, MPI_STATUS_IGNORE );

        sum += temp_val;

        prefixes.push_back( sum );

        if( i % ( comm_size - 1 ) == 0 ){

            temp_val = elements[ i + my_rank ];
        }

        else{

            continue;
        }
    }

    return prefixes;
}

int main( ){

    int my_rank{ 0 }, comm_size{ 0 };

    MPI_Init( nullptr, nullptr );
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );

    std::vector<int> test{ 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120 };

    //std::vector<int> partial_sum = prefix_sum( my_rank, comm_size, my_rank, MPI_COMM_WORLD );
    std::vector<int> partial_sum = generalized_prefix_sum( test, my_rank, comm_size, MPI_COMM_WORLD );

    if( my_rank == 0 ){

        for( auto i = 0; i < partial_sum.size( ); ++i ){

            printf( "%d ", partial_sum[ i ] );
        }

        printf( "\n" );
    }

    MPI_Finalize( );
}