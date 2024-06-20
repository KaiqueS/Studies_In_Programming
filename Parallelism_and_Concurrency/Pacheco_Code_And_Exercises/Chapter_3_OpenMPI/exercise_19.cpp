#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>

// 3.19 MPI_Type_indexed can be used to build a derived datatype from arbitrary array elements. Its syntax is

// int MPI_Type_indexed ( int count, int array_of_blocklengths[ ], int array_of_displacements[ ], MPI_Datatype old_mpi_t, MPI_Datatype∗ new_mpi_t_p ) ;

// Unlike MPI_Type_create_struct, the displacements are measured in units of old_mpi_t—not bytes. Use MPI_Type_indexed
// to create a derived datatype that corresponds to the upper triangular part of a square matrix.

// Process 0 should read in an n × n matrix as a one-dimensional array, create the derived datatype, and send the
// upper triangular part with a single call to MPI_Send. Process 1 should receive the upper triangular part with
// a single call to MPI_Recv, and then print the data it received.

void Read_Matrix( int row, MPI_Datatype type, MPI_Comm comm ){

    std::random_device device;
    std::uniform_real_distribution<double> dist( -row, row );
    std::mt19937_64 rng( device( ) );

    std::vector<std::vector<double>> matrix( row, std::vector<double>( row ) );

    for( auto i = 0; i < matrix.size( ); ++i ){

        for( auto j = 0; j < matrix[ i ].size( ); ++j ){

            matrix[ i ][ j ] = dist( rng );
        }
    }

    
}

void Print_Data( MPI_Datatype type, MPI_Comm comm ){


}

int main( ){

    int rank{ 0 }, comm_size{ 0 };

    MPI_Init( nullptr, nullptr );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );

    MPI_Finalize( );
}