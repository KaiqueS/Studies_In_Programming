#include <pthread.h>
#include <iostream>
#include <vector>
#include <random>
#include <cstring>
#include <cassert>

int thread_count{ 0 };
int m{ 0 }, n{ 0 };

std::vector<std::vector<double>> A{ };
std::vector<double> x{ }, y{ };

void fill_matrix( std::vector<std::vector<double>>& matrix, int rows, int columns ){

    std::random_device device;
    std::uniform_real_distribution<double> dist( -( rows * columns ), ( rows * columns ) );
    std::mt19937_64 rng( device( ) );

    for( auto i = 0; i < rows; ++i ){

        for( auto j = 0; j < columns; ++j ){

            matrix[ i ][ j ] = dist( rng );
        }
    }
}

void fill_vector( std::vector<double>& vetor, int elements ){

    std::random_device device;
    std::uniform_real_distribution<double> dist( -elements, elements );
    std::mt19937_64 rng( device( ) );

    for( auto i = 0; i < elements; ++i ){

        vetor[ i ] = dist( rng );
    }
}

void* Pth_math_vect( void* rank ){

    //long my_rank = ( long ) rank;
    //long* my_rank = static_cast<long>( rank ); // why is static_cast<void*>( rank ) wrong?
    long my_rank{ };
    assert( sizeof my_rank == sizeof rank );
    std::memcpy( &my_rank, &rank, sizeof my_rank );

    int local_m = std::ceil( double( m / thread_count ) );
    int my_first_row = my_rank * local_m;
    int my_last_row = ( ( my_rank + 1 ) * local_m ) - 1;

    for( auto i = my_first_row; i <= my_last_row; ++i ){

        for( auto j = 0; j < n; ++j ){

            y[ i ] += A[ i ][ j ] * x[ j ];
        }
    }

    return nullptr;
}

int main( int argc, char* argv[ ] ){

    thread_count = strtol( argv[ 1 ], nullptr, 10 );

    pthread_t* thread_handles = new pthread_t( thread_count );

    printf( "Enter m and n: \n" );
    scanf( "%d %d", &m, &n );

    A = std::vector<std::vector<double>>( m, std::vector<double>( n, 0 ) );
    x = std::vector<double>( n, 0 );
    y = std::vector<double>( n, 0 );

    fill_matrix( A, m, n );
    fill_vector( x, n );

    for( long thread = 0; thread < thread_count; ++thread ){

        pthread_create( &thread_handles[ thread ], nullptr, Pth_math_vect, ( void* ) thread );
        // pthread_create( &thread_handles[ thread ], nullptr, Pth_math_vect, static_cast<void*>( &thread ) ); // why is static_cast<void*>( thread ) wrong?
    }

    for( long thread = 0; thread < thread_count; ++thread ){

        pthread_join( thread_handles[ thread ], nullptr );
    }

    for( auto i = 0; i < y.size( ); ++i ){

        std::cout << y[ i ] << " ";

        if( i == y.size( ) - 1 ){

            std::cout << "\n";
        }
    }

    delete thread_handles;
}