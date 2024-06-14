#include <mpi.h>
#include <iostream>
#include <vector>

// 3.17 MPI_Type_contiguous can be used to build a derived datatype from a collection of contiguous elements in an array. Its syntax is

// int MPI_Type_contiguous( int count, MPI_Datatype old_mpi_t, MPI_Datatype∗ new_mpi_t_p );

// Modify the Read_vector and Print_vector functions so that they use an MPI datatype created by a call to MPI_Type_contiguous and
// a count argument of 1 in the calls to MPI_Scatter and MPI_Gather.

class mpi_input{

public:

    mpi_input( ){ }
    ~mpi_input( ){

        MPI_Type_free( &input_mpi_t );
    }

    MPI_Datatype& datatype( ){ return input_mpi_t; }

    void Build_mpi_type( double* a_p, double* b_p, int* n_p ){

        int array_of_blocklengths[ 3 ] = { 1, 1, 1 };

        MPI_Datatype array_of_types[ 3 ] = { MPI_DOUBLE, MPI_DOUBLE, MPI_INT };

        MPI_Aint a_addr, b_addr, n_addr;
        MPI_Aint array_of_displacements[ 3 ] = { 0 };

        MPI_Get_address( a_p, &a_addr );
        MPI_Get_address( b_p, &b_addr );
        MPI_Get_address( n_p, &n_addr );

        array_of_displacements[ 1 ] = b_addr - a_addr;
        array_of_displacements[ 2 ] = n_addr - a_addr;

        MPI_Type_create_struct( 3, array_of_blocklengths, array_of_displacements, array_of_types, &input_mpi_t );
        MPI_Type_commit( &input_mpi_t );
    }

private:

    MPI_Datatype input_mpi_t;
};

void Read_vector( std::vector<double>& local_a, int n, int my_rank, MPI_Comm comm ){

    std::vector<double> a{ };

    MPI_Datatype newtype;

    MPI_Type_contiguous( local_a.size( ), MPI_DOUBLE, &newtype );
    MPI_Type_commit( &newtype );

    if( my_rank == 0 ){

        a.resize( n );

        for( int i = 0; i < n; ++i ){
        
            a[ i ] = static_cast<double>( i );
        }

        MPI_Scatter( a.data( ), 1, newtype, local_a.data( ), 1, newtype, 0, comm );

        a.clear( );
    }

    else{

        // This is interesting: a[] is empty when any process != 0 accesses it. I THINK that MPI_Scatter
        // searches for a[ ] in comm, and the uses the processes' rank to find the correct portion of a[ ]
        // that will be stored in local_a[ ]
        MPI_Scatter( a.data( ), 1, newtype, local_a.data( ), 1, newtype, 0, comm );
    }

    MPI_Type_free( &newtype );
}

void Print_vector( std::vector<double>& local_b, int n, int my_rank, MPI_Comm comm ){

    std::vector<double> b{ };

    MPI_Datatype newtype;

    MPI_Type_contiguous( local_b.size( ), MPI_DOUBLE, &newtype );
    MPI_Type_commit( &newtype );

    if( my_rank == 0 ){
        
        b.resize( n );

        MPI_Gather( local_b.data( ), 1, newtype, b.data( ), 1, newtype, 0, comm );

        for( std::vector<double>::size_type i = 0; i < b.size( ); ++i ){

            std::cout << b[ i ] << " ";
        }

        printf( "\n" );

        b.clear( );
    }

    else{

        MPI_Gather( local_b.data( ), 1, newtype, b.data( ), 1, newtype, 0, comm );
    }

    MPI_Type_free( &newtype );
}

int main( ){

    int rank{ 0 }, comm_size{ 0 };

    MPI_Init( nullptr, nullptr );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );

    int size{ 0 };

    if( rank == 0 ){

        printf( "Enter the size of the vector: \n" );
        scanf( "%d", &size );
    }

    MPI_Bcast( &size, 1, MPI_INT, 0, MPI_COMM_WORLD );

    std::vector<double> local_vector( size / comm_size );

    Read_vector( local_vector, size, rank, MPI_COMM_WORLD );

    Print_vector( local_vector, size, rank, MPI_COMM_WORLD );

    MPI_Finalize( );
}