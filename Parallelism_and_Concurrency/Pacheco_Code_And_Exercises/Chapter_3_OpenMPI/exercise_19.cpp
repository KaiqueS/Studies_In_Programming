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

std::vector<double> Build_Matrix( int row ){

    std::random_device device;
    std::uniform_real_distribution<double> dist( -( row * row ), ( row * row ) );
    std::mt19937_64 rng( device( ) );

    std::vector<double> matrix( row * row );

    for( auto i = 0; i < matrix.size( ); ++i ){
    
        matrix[ i ] = dist( rng );    
    }

    return matrix;
}

std::vector<int> Block_length( int row ){

    std::vector<int> Blocks( row );

    for( auto i = 0; i < Blocks.size( ); ++i ){

        Blocks[ i ] = row - i;
    }

    return Blocks;
}

int count( std::vector<int>& blocks ){

    int elements{ 0 };

    for( auto i = 0; i < blocks.size( ); ++i ){

        elements += blocks[ i ];
    }

    return elements;
}

/// NOTE: the amount of displacements MUST match the amount of blocks. Even if it happens
///       that one displacement == 0!!!!!!!
std::vector<int> Displacements( int row ){

    std::vector<int> result{ };

    for( auto i = 0; i < row; ++i ){

        //result.push_back( ( row * i ) + i );
        result.push_back( i );
    }

    return result;
}

void Read_Matrix( int row, MPI_Comm comm ){

    std::vector<double> matrix = Build_Matrix( row );
    std::vector<int> blocks = Block_length( row );
    std::vector<int> displace = Displacements( row ); // CHECK NOTE ON DISPLACEMENTS
    int total = count( blocks );

    MPI_Datatype type;
    MPI_Type_indexed( row, blocks.data( ), displace.data( ), MPI_DOUBLE, &type ); // // CHECK NOTE ON DISPLACEMENTS
    MPI_Type_commit( &type );

    MPI_Send( &total, 1, MPI_INT, 1, 0, comm );
    MPI_Send( matrix.data( ), 1, type, 1, 0, comm );

    MPI_Type_free( &type );
}

void Print_Data( MPI_Comm comm ){

    int amount{ 0 };
    MPI_Recv( &amount, 1, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE );

    std::vector<double> output( amount );

    MPI_Recv( output.data( ), output.size( ), MPI_DOUBLE, 0, 0, comm, MPI_STATUS_IGNORE );

    for( auto i = 0; i < output.size( ); ++i ){

        std::cout << output[ i ] << " ";
    }

    printf( "\n" );
}

int main( ){

    int rank{ 0 }, comm_size{ 0 };

    MPI_Init( nullptr, nullptr );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );

    if( rank == 0 ){

        int size{ 0 };

        printf( "Enter the dimensions of the matrix: \n" );
        scanf( "%d", &size );

        Read_Matrix( size, MPI_COMM_WORLD );
    }

    else{

        Print_Data( MPI_COMM_WORLD );
    }

    MPI_Finalize( );
}