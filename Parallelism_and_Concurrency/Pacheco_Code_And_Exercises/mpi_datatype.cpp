#include <iostream>
#include <string>
#include <stdio.h>
#include <functional>
#include <mpi.h>
#include <vector>
#include <unistd.h>

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

double Trap( double left_end, double right_end,  int trap_count, double base_len, std::function<double( double )> f ){

    double estimate{ }, x{ };
    int i{ };

    estimate = ( f( left_end ) + f( right_end ) ) / 2.0;

    for( i = 1; i <= ( trap_count - 1 ); ++i ){

        x = left_end + i * base_len;

        estimate += f( x );
    }

    estimate = ( estimate * base_len );

    return estimate;
}

double quadratic( double x ){

    double square = ( x * x );

    return square;
}

void Get_input( int my_rank, int comm_sz, double* a_p, double* b_p, int* n_p ){

    mpi_input my_type;

    my_type.Build_mpi_type( a_p, b_p, n_p );

    if( my_rank == 0 ){

        printf( "Enter a, b, and n\n" );
        scanf( "%lf %lf %d", a_p, b_p, n_p );
    }

    /// MPI_Bcast sends to ALL processes messages from ONE process, here 0
    // COMMENT: the derived datatype is correctly broadcasted. But, the problem is - I cannot retrieve
    // the information I want from within it - DONE, just follow the author's code
    // EXPLANATION: here, a_p serves only as a "binding" for the shared memory-space storing {a_p, b_p, n_p}.
    //              Now, my guessing: the datatype creates a pack composed by some basic types, their addresses
    //              in memory, and their displacement. Broadcast, then, allocates in a shared space, addresses
    //              for this pack, whilst remembering its format( addresses + types + displacement ). Even though
    //              only the master process reads input, all processes must provide their own copies of the target
    //              values( a_p_, b_p, and n_p ), and build their own copies of the derived datatype, this datatype
    //              binds the processes' target values to the packed structure, and, when the process executes its
    //              call to the MPI_Bcast, all the packed target values get the values broadcasted by the master process,
    //              even though MPI_Bcast is called only with a_p. Because a_p now is only a binding that MPI_Bcast uses
    //              to access the first( here ) term of the pack( which contains a_p, b_p, and n_p ). When the call ends,
    //              this pack is unpacked and destructed, without destructing the target values.
    // TL;DR: a_p is a bind to a pack containing {a_p, b_p, n_p}, instead of a single variable. This is caused by the
    //        derived datatype.
    MPI_Bcast( a_p, 1, my_type.datatype( ), 0, MPI_COMM_WORLD );
}

void Parallel_vector_sum( std::vector<double> local_x, std::vector<double> local_y, std::vector<double>& local_z, int local_n ){

    for( int local_i = 0; local_i < local_n; ++local_i ){

        local_z[ local_i ] = local_x[ local_i ] + local_y[ local_i ];
    }
}

void Read_vector( std::vector<double>& local_a, int local_n, int n, int my_rank, MPI_Comm comm ){

    std::vector<double> a{ };

    if( my_rank == 0 ){

        a.resize( n );

        for( int i = 0; i < n; ++i ){
        
            a[ i ] = static_cast<double>( i );
        }

        MPI_Scatter( a.data( ), local_n, MPI_DOUBLE, local_a.data( ), local_n, MPI_DOUBLE, 0, comm );

        a.clear( );
    }

    else{

        // This is interesting: a[] is empty when any process != 0 accesses it. I THINK that MPI_Scatter
        // searches for a[ ] in comm, and the uses the processes' rank to find the correct portion of a[ ]
        // that will be stored in local_a[ ]
        MPI_Scatter( a.data( ), local_n, MPI_DOUBLE, local_a.data( ), local_n, MPI_DOUBLE, 0, comm );
    }
}

void Print_vector( std::vector<double>& local_b, int local_n, int n, int my_rank, MPI_Comm comm ){

    std::vector<double> b{ };

    if( my_rank == 0 ){
        
        b.resize( n );

        MPI_Gather( local_b.data( ), local_n, MPI_DOUBLE, b.data( ), local_n, MPI_DOUBLE, 0, comm );

        for( std::vector<double>::size_type i = 0; i < b.size( ); ++i ){

            std::cout << b[ i ] << " ";
        }

        std::cout << "\n";

        b.clear( );
    }

    else{

        MPI_Gather( local_b.data( ), local_n, MPI_DOUBLE, b.data( ), local_n, MPI_DOUBLE, 0, comm );
    }
}

void Mat_vect_mult( std::vector<std::vector<double>> local_A, std::vector<double> local_x, std::vector<double>& local_y, int local_m, int n, int local_n, MPI_Comm comm ){

    std::vector<double> x{ };

    int local_i{ }, local_ok{ 1 };

    x.resize( n );

    // Here, all processes put they local_x[] elements into x[]. With this, all processes can
    // access through x[] all the local_x[] elements from other processes.
    MPI_Allgather( local_x.data( ), local_n, MPI_DOUBLE, x.data( ), local_n, MPI_DOUBLE, comm );

    for( std::vector<double>::size_type local_i = 0; local_i < local_m; ++local_i ){

        local_y[ local_i ] = 0.0;

        for( std::vector<double>::size_type j = 0; j < n; ++j ){

            local_y[ local_i ] += local_A[ local_i ][ j ] * x[ j ];
        }
    }

    x.clear( );
}

void test_allgather( ){

    int my_rank{ 0 }, comm_sz{ 0 }, n{ 0 }, local_n{ 0 };

    n = ( 100 * comm_sz );

    local_n = ( n / comm_sz );

    std::vector<double> local_a( local_n, 0 );
    std::vector<double> local_b( local_n, 0 );
    std::vector<double> result( local_n, 0 );

    std::vector<std::vector<double>> matrix( local_n, std::vector<double>( local_n, 0 ) );

    // Parallel_vector_sum( a, b, result, 10 );

    if( my_rank % 2 == 0 ){

        Read_vector( local_a, local_n, n, my_rank, MPI_COMM_WORLD );
        Read_vector( local_b, local_n, n, my_rank, MPI_COMM_WORLD );

        for( auto i = 0; i < matrix.size( ); ++i ){

            matrix[ i ] = local_a;
        }
        //Parallel_vector_sum( local_a, local_b, result, local_n );
        Mat_vect_mult( matrix, local_b, result, local_n, n, local_n, MPI_COMM_WORLD );
        //MPI_Gather( local_a.data( ), local_n , MPI_DOUBLE, result.data( ), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD );
    }

    else{

        Read_vector( local_a, local_n, n, my_rank, MPI_COMM_WORLD );
        Read_vector( local_b, local_n, n, my_rank, MPI_COMM_WORLD );

        for( auto i = 0; i < matrix.size( ); ++i ){

            matrix[ i ] = local_a;
        }
        //Parallel_vector_sum( local_a, local_b, result, local_n );
        Mat_vect_mult( matrix, local_b, result, local_n, n, local_n, MPI_COMM_WORLD );
        //MPI_Gather( local_a.data( ), local_n, MPI_DOUBLE, result.data( ), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD );
    }

    Print_vector( result, local_n, n, my_rank, MPI_COMM_WORLD );

}

int main( void ){

    int my_rank{ 0 }, comm_sz{ 0 }, n{ 0 }, local_n{ 0 };

    double a{ 0.0 }, b{ 0.0 }, h{ 0.0 }, local_a{ 0.0 }, local_b{ 0.0 };
    double local_int{ 0.0 }, total_int{ 0.0 };

    MPI_Init( NULL, NULL );
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &comm_sz );

    //mpi_input send_data{ };

    //Get_input_book( my_rank, comm_sz, &a, &b, &n );
    Get_input( my_rank, comm_sz, &a, &b, &n );

    //h = ( send_data.get_b( ) - send_data.get_a( ) ) / send_data.get_n( );
    h = ( b - a ) / n;

    //local_n = send_data.get_n( ) / comm_sz;
    local_n = n / comm_sz;

    //local_a = send_data.get_a( ) + my_rank * local_n * h;
    local_a = a + my_rank * local_n * h;

    local_b = local_a + local_n * h;
    local_int = Trap( local_a, local_b, local_n, h, quadratic );

    /// MPI_Allreduce is a form of Collective Communication: it gets the local_int sent by each process,
    /// apply a function to them( here MPI_SUM ), and stores the result in total_int
    MPI_Allreduce( &local_int, &total_int, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

    if( my_rank == 0 ){

        printf( "With n = %d trapezoids, our estimate\n", n );
        printf( "of the integral from %f to %f = %.15e\n", a, b, total_int );
    }

    MPI_Finalize( );

    return 0;
}