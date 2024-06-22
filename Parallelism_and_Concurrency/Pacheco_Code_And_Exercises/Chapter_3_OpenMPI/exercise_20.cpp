#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>

// The functions MPI_Pack and MPI_Unpack provide an alternative to derived datatypes for grouping data. MPI_Pack copies the data to be sent, one block at a time,
// into a user-provided buffer. The buffer can then be sent and received. After the data is received, MPI_Unpack can be used to unpack it from the receive buffer.
// The syntax of MPI_Pack is

// int MPI_Pack( void* in_buf, int in_buf_count, MPI_Datatype datatype, void* pack_buf, int pack_buf_sz, int position_p, MPI_Comm comm );

// When MPI_Pack is called, position should refer to the first available slot in pack_buf. When MPI_Pack returns, it refers to the first available slot after the data
// that was just packed

// Now the other processes can unpack the data using MPI_Unpack:

// int MPI_Unpack( void∗ pack_buf, int pack_buf_sz, int∗ position_p, void∗ out_buf, int out_buf_count, MPI_Datatype datatype, MPI_Comm comm );

// Write another Get_input function for the trapezoidal rule program. This one should use MPI_Pack on process 0, and MPI_Unpack on the other processes.

/// ANSWER:

void Get_Input( int rank, int comm_size, double a_p, double b_p, int n_p, MPI_Comm comm ){

    int dest{ };

    

    MPI_Datatype type;
    MPI_Pack(  )

    if( rank == 0 ){

        printf( "Enter a, b, and n\n" );

        scanf( "%lf %lf %d", &a_p, &b_p, &n_p );

        for( dest = 1; dest < comm_size; ++dest ){

            MPI_Send( a_p, 1, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD );
            MPI_Send( b_p, 1, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD );
            MPI_Send( n_p, 1, MPI_INT, dest, 0, MPI_COMM_WORLD );
        }
    }

    else{

        MPI_Recv( a_p, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        MPI_Recv( b_p, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        MPI_Recv( n_p, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
    }
}

int main( ){

    int rank{ 0 }, comm_size{ 0 };

    MPI_Init( nullptr, nullptr );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );



    MPI_Finalize( );
}