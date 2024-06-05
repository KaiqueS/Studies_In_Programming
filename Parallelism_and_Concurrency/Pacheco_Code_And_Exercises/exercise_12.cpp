#include <mpi.h>
#include <iostream>
#include <vector>

/*
3.12 An alternative to a butterfly-structured allreduce is a “ring-pass” structure. In a ring-pass, if there
are p processes, each process q sends data to process q + 1, except that process p − 1 sends data to process
0. This is repeated until each process has the desired result. Thus we can implement allreduce with the following code:

    sum = temp_val = my_val;

    for( i = 1 ; i < p ; i ++ ){
        
        MPI_Sendrecv_replace( &temp_val, 1, MPI_INT, dest, sendtag, source, recvtag, comm, &status );
        
        sum += temp_val;
    }
*/

//  a.  Write an MPI program that implements this algorithm for allreduce. How does its performance compare to the
//      butterfly-structured allreduce?

void MPI_Ring_Allreduce( int my_rank, int comm_size, int my_val, int& result, MPI_Comm comm ){

    int sum{ my_val }, temp_val{ my_val };

    for( auto i = 1; i < comm_size; ++i ){

        // This is wrong: whenever i + 1 > comm_size, this modulo will return an invalid receiver index
        MPI_Sendrecv_replace( &temp_val, 1, MPI_INT, ( ( i + 1 ) % comm_size ), 0, i, 0, comm, MPI_STATUS_IGNORE  );

        sum += temp_val;
    }

    result = sum;
}

// b.  Modify the MPI program you wrote in the first part so that it implements prefix sums.


int main( ){

    int my_rank{ 0 }, comm_size{ 0 };

    MPI_Init( nullptr, nullptr );
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );

    int result{ 0 };

    MPI_Ring_Allreduce( my_rank, comm_size, my_rank, result, MPI_COMM_WORLD );

    if( my_rank == 0 ){

        printf( "%d", result );

        printf( "\n" );
    }

    MPI_Finalize( );
}