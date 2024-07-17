#include <pthread.h>
#include <iostream>
#include <vector>
#include <random>

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

    long* my_rank = static_cast<long*>( rank );

    int local_m = m / thread_count;
    int my_first_row = *my_rank * local_m;
    int my_last_row = ( *my_rank + 1 ) * local_m - 1;

    for( auto i = my_first_row; i <= my_last_row; ++i ){

        y[ i ] = 0.0;

        for( auto j = 0; j < n; ++j ){

            y[ i ] = A[ i ][ j ] * x[ j ];
        }
    }

    return nullptr;
}

int main( ){

    printf( "Enter m and n: \n" );
    scanf( "%d %d", m, n );

    A.resize( m );
    x.resize( n );
    y.resize( n );

    for( auto i = 0; i < A.size( ); ++i ){

        A[ i ].resize( n );
    }

    fill_matrix( A, m, n );
    fill_vector( x, n );

    

}