#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>

// 3.18 MPI_Type_vector can be used to build a derived datatype from a collection of blocks of elements in an array,
// as long as the blocks all have the same size and they’re equally spaced. Its syntax is

    // int MPI_Type_vector( int count, int blocklength, int stride, MPI_Datatype old_mpi_t, MPI_Datatype* new_mpi_t_p );

// Write Read_vector and Print_vector functions that will allow process 0 to read and print, respectively, a vector with
// a block-cyclic distribution. But beware! Do not use MPI_Scatter or MPI_Gather.

// Just use a loop of sends on process 0 in Read_vector and a loop of receives on process 0 in Print_vector. The other
// processes should be able to complete their calls to Read_vector and Print_vector with a single call to MPI_Recv and
// MPI_Send, respectively. The communication on process 0 should use a derived datatype created by MPI_Type_vector. The
// calls on the other processes should just use the count argument to the communication function, since they’re receiving/sending
// elements that they will store in contiguous array locations

/// ANSWER:

std::vector<double> build_vector( int vec_size ){

    std::vector<double> values( vec_size );

    std::random_device rnd_dev;
    std:: uniform_real_distribution<double> dist( -vec_size, vec_size );
    std::mt19937_64 rng( rnd_dev( ) );

    for( std::vector<double>::size_type i = 0; i < values.size( ); ++i ){

        values[ i ] = dist( rng );
    }

    return values;
}

void Read_vector( std::vector<double>& my_block, int rank, int comm_size, int vec_size, int num_of_blocks, int stride, int block_length, MPI_Comm comm ){

    MPI_Datatype vect_mpi_t;

    MPI_Type_vector( num_of_blocks, block_length, stride, MPI_DOUBLE, &vect_mpi_t );
    MPI_Type_commit( &vect_mpi_t );

    if( rank == 0 ){

        std::vector<double> elements{ };
        elements = build_vector( vec_size );

        for( auto i = 0; i < comm_size; ++i ){

            MPI_Send( elements.data( ), 1, vect_mpi_t, i, 0, comm );
        }
    }

    // MPI_Bcast( vect_mpi_t, 1, vect_mpi_t, 0, comm ); - THIS WON'T WORK: since processes != 0 did not call type_commit nor type_vector,
    //                                                    their instances of vect_mpi_t are not appropriately formatted in the memory buffer.
    //                                                    Thus, the call to Bcast will be writing to an unformatted buffer, causing segfault

    my_block.resize( block_length );
    
    MPI_Recv( my_block.data( ), 1, vect_mpi_t, 0, 0, comm, MPI_STATUS_IGNORE );
    
    MPI_Type_free( &vect_mpi_t );
}

void Print_vector( std::vector<double>& my_block, int rank, int comm_size, int vec_size, int num_of_blocks, int stride, int block_length, MPI_Comm comm ){

    MPI_Datatype vect_mpi_t;

    MPI_Type_vector( num_of_blocks, block_length, stride, MPI_DOUBLE, &vect_mpi_t );
    MPI_Type_commit( &vect_mpi_t );

    MPI_Send( my_block.data( ), 1, vect_mpi_t, 0, 0, comm );
    
    if( rank == 0 ){

        std::vector<double> output( ( my_block.size( ) * comm_size ) );

        for( auto i = 0; i < comm_size; ++i ){

            MPI_Recv( output.data( ), 1, vect_mpi_t, i, 0, comm, MPI_STATUS_IGNORE );      
        }

        for( auto i = 0; i < output.size( ); ++i ){

            std::cout << output[ i ] << " ";
        }

        printf( "\n" );
    }

    MPI_Type_free( &vect_mpi_t );
}

int main( ){

    int rank{ 0 }, comm_size{ 0 };

    MPI_Init( nullptr, nullptr );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );

    int vec_size{ 0 }, num_of_blocks{ 0 }, stride{ 0 }, block_length{ 0 };

    if( rank == 0 ){

        printf( "Enter the size of the vector: \n" );
        scanf( "%d", &vec_size );

        printf( "Enter the amount of blocks, their size, and its stride: \n" );
        scanf( "%d %d %d", &num_of_blocks, &block_length, &stride );
    }

    MPI_Bcast( &vec_size, 1, MPI_INT, 0, MPI_COMM_WORLD );
    MPI_Bcast( &num_of_blocks, 1, MPI_INT, 0, MPI_COMM_WORLD );
    MPI_Bcast( &stride, 1, MPI_INT, 0, MPI_COMM_WORLD );
    MPI_Bcast( &block_length, 1, MPI_INT, 0, MPI_COMM_WORLD );

    std::vector<double> my_block{ };

    Read_vector( my_block, rank, comm_size, vec_size, num_of_blocks, stride, block_length, MPI_COMM_WORLD );
    Print_vector( my_block, rank, comm_size, vec_size, num_of_blocks, stride, block_length, MPI_COMM_WORLD );

    MPI_Finalize( );
}