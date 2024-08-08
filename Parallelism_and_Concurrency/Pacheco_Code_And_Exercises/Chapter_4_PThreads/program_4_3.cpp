#include <pthread.h>
#include <iostream>
#include <cstring>
#include <cassert>

int thread_count{ 0 };
int n{ 0 };

double pi{ 0 };

void* Thread_Sum( void* rank ){

    long my_rank{ 0 };
    assert( sizeof my_rank == sizeof rank );
    std::memcpy( &my_rank, &rank, sizeof my_rank );

    double factor{ 0 };
    double sum{ 0.0 };

    long long my_n = n / thread_count;
    long long my_first_i = my_n * my_rank;
    long long my_last_i = my_first_i + my_n;

    if( my_first_i % 2 == 0 ){

        factor = 1.0;
    }

    else{

        factor = -1.0;
    }

    for( long long i = my_first_i; i < my_last_i; ++i, factor = -factor ){

        sum += factor / ( 2 * i + 1 );
    }

    pi = 4.0 * sum;

    return nullptr;
}

int main( int argc, char* argv[ ] ){

    thread_count = strtol( argv[ 1 ], nullptr, 10 );

    pthread_t* thread_handles = new pthread_t( thread_count );

    printf( "Enter n: \n" );
    scanf( "%d", &n );

    for( long thread{ 0 }; thread < thread_count; ++thread ){

        pthread_create( &thread_handles[ thread ], nullptr, Thread_Sum, ( void* ) thread );
    }

    for( long thread{ 0 }; thread < thread_count; ++thread ){

        pthread_join( thread_handles[ thread ], nullptr );
    }

    printf( "%lf \n", pi );

    delete thread_handles;
}