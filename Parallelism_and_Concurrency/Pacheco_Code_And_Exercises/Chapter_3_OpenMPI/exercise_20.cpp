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

    char buffer[ 100 ];

    if( rank == 0 ){

        int position{ 0 };

        printf( "Enter a, b, and n\n" );

        scanf( "%lf %lf %d", &a_p, &b_p, &n_p );

        MPI_Pack( &a_p, 1, MPI_DOUBLE, buffer, 100, &position, comm );
        MPI_Pack( &b_p, 1, MPI_DOUBLE, buffer, 100, &position, comm );
        MPI_Pack( &n_p, 1, MPI_INT, buffer, 100, &position, comm );

        MPI_Bcast( buffer, 100, MPI_PACKED, 0, comm );
    }

    else{

        MPI_Bcast( buffer, 100, MPI_PACKED, 0, comm );

        int position{ 0 };

        MPI_Unpack( buffer, 100, &position, &a_p, 1, MPI_DOUBLE, comm );
        MPI_Unpack( buffer, 100, &position, &b_p, 1, MPI_DOUBLE, comm );
        MPI_Unpack( buffer, 100, &position, &n_p, 1, MPI_INT, comm );
    }
}

int main( ){

    int rank{ 0 }, comm_size{ 0 };

    MPI_Init( nullptr, nullptr );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );

    double a_p{ 0 }, b_p{ 0 };
    int n_p{ 0 };

    Get_Input( rank, comm_size, a_p, b_p, n_p, MPI_COMM_WORLD );

    MPI_Finalize( );
}