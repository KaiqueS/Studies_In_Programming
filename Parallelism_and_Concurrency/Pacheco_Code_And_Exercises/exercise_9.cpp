#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>

// 3.9 Write an MPI program that implements multiplication of a vector by a scalar
// and dot product. The user should enter two vectors and a scalar, all of which
// are read in by process 0 and distributed among the processes. The results are
// calculated and collected onto process 0, which prints them. You can assume
// that n, the order of the vectors, is evenly divisible by comm_sz.

/// ANSWER: to make testing faster, I am not implementing user input. Instead, I use
///         a RNG to fill the vectors. The user only sets the size of the vectors and
///         the scalar.

void program_input( double& scalar, int& size ){

    printf( "Enter a scalar and the vectors' size: " );
    scanf( "%lf %d", &scalar, &size );
}

void mult_vector_scalar( double scalar, std::vector<double>& vec ){

    for( std::vector<double>::size_type i = 0; i < vec.size( ); ++i ){

        vec[ i ] *= scalar;
    }
}

double dot_product( std::vector<double>& left, std::vector<double>& right ){

    double sum{ 0 };

    for( std::vector<double>::size_type i = 0; i < left.size( ); ++i ){

        sum += left[ i ] * right[ i ];
    }

    return sum;
}

void create_shared_data( std::vector<double>& left, std::vector<double>& right ){

    double scalar{ 0 };
    int size{ 0 };

    program_input( scalar, size );

    left.resize( size );
    right.resize( size );

    std::random_device left_rand, right_rand;
    std:: uniform_real_distribution<double> dist( -size, size );
    std::mt19937_64 left_rng( left_rand( ) ), right_rng( right_rand( ) );

     for( std::vector<double>::size_type i = 0; i < size; ++i ){

        left[ i ] = dist( left_rng );
        right[ i ] = dist( right_rng );
     }
}

void share_data( int my_rank, int comm_sz, std::vector<double>& left, std::vector<double>& right, MPI_Comm comm ){

    std::vector<double> holder_l{ }, holder_r{ };

    int buffsize{ 0 };

    if( my_rank == 0 ){

        create_shared_data( holder_l, holder_r );

        buffsize = ( holder_l.size( ) / comm_sz ) + ( holder_l.size( ) % comm_sz );
        
        left.resize( buffsize );
        right.resize( buffsize );

        MPI_Bcast( &buffsize, 1, MPI_INT, 0, comm );

        MPI_Scatter( holder_l.data( ), buffsize, MPI_DOUBLE, left.data( ), buffsize, MPI_DOUBLE, 0, comm );
        MPI_Scatter( holder_r.data( ), buffsize, MPI_DOUBLE, right.data( ), buffsize, MPI_DOUBLE, 0, comm );

        holder_l.clear( );
        holder_r.clear( );
    }

    else{

        MPI_Bcast( &buffsize, 1, MPI_INT, 0, comm );

        left.resize( buffsize );
        right.resize( buffsize );

        MPI_Scatter( holder_l.data( ), buffsize, MPI_DOUBLE, left.data( ), buffsize, MPI_DOUBLE, 0, comm );
        MPI_Scatter( holder_r.data( ), buffsize, MPI_DOUBLE, right.data( ), buffsize, MPI_DOUBLE, 0, comm );
    }
}

int main( ){

    int comm_sz{ 0 }, my_rank{ 0 };

    MPI_Init( nullptr, nullptr );
    MPI_Comm_size( MPI_COMM_WORLD, &comm_sz );
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

    std::vector<double> left{ }, right{ };
    
    share_data( my_rank, comm_sz, left, right, MPI_COMM_WORLD );
    
    double dot = dot_product( left, right );

    double total_dot{ 0 };

    MPI_Allreduce( &dot, &total_dot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

    if( my_rank == 0 ){

        printf( "\nThe dot product is: %f", total_dot );
    }

    MPI_Finalize( );
}