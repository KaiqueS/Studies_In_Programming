#include <iostream>
#include <string>
#include <stdio.h>
#include <functional>
#include <mpi.h>
#include <vector>
#include <unistd.h>

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

    if( my_rank == 0 ){

        printf( "Enter a, b, and n\n" );

        scanf( "%lf %lf %d", a_p, b_p, n_p );
    }

    /// MPI_Bcast sends to ALL processes messages from ONE process, here 0
    MPI_Bcast( a_p, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD );
    MPI_Bcast( b_p, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD );
    MPI_Bcast( n_p, 1, MPI_INT, 0, MPI_COMM_WORLD );
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

// COMMENT: finally found the problems
//          1- to properly use a std::vector as a receiving buffer, I need to either pass vector.data(),
//             or pass vector[ 0 ]. .data() is preferred here <- THIS SOLVES THE PROBLEM!
//                  -> I can also use &*vector.begin( ), &vector[ 0 ], but .data() is safer and cleaner
//          2- since I'm handling vector<double>, and not double itself, MPI_DOUBLE sends and receives
//             incorrect values, because it is reinterpreting a pointer to a double as a double. To fix
//             this, I need to create a MPI Type wrapping a vector of doubles <- See 1-.!

int main( void ){

    int my_rank{ 0 }, comm_sz{ 0 }, n{ 0 }, local_n{ 0 };

    MPI_Init( NULL, NULL ); 
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &comm_sz );

    n = ( 100 * comm_sz );

    local_n = ( n / comm_sz );

    std::vector<double> local_a( local_n, 0 );
    std::vector<double> local_b( local_n, 0 );
    std::vector<double> result( local_n, 0 );

    // Parallel_vector_sum( a, b, result, 10 );

    if( my_rank % 2 == 0 ){

        Read_vector( local_a, local_n, n, my_rank, MPI_COMM_WORLD );
        Read_vector( local_b, local_n, n, my_rank, MPI_COMM_WORLD );

        Parallel_vector_sum( local_a, local_b, result, local_n );
        //MPI_Gather( local_a.data( ), local_n , MPI_DOUBLE, result.data( ), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD );
    }

    else{

        Read_vector( local_a, local_n, n, my_rank, MPI_COMM_WORLD );
        Read_vector( local_b, local_n, n, my_rank, MPI_COMM_WORLD );

        Parallel_vector_sum( local_a, local_b, result, local_n );
        //MPI_Gather( local_a.data( ), local_n, MPI_DOUBLE, result.data( ), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD ); // This might be similar to how MPI_Scatter works: here, result[ ], even though process-local,
                                                                                                                      // is handled by MPI only as an identifier, which it uses to find the correct buffer.
    }

    Print_vector( result, local_n, n, my_rank, MPI_COMM_WORLD );

    MPI_Finalize( );

    return 0;
}