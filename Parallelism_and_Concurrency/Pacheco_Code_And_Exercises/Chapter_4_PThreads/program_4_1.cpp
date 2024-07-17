#include <pthread.h>
#include <iostream>

int thread_count{ 0 };

void* Hello( void* rank ){

    long* my_rank = static_cast<long*>( rank );

    printf( "Hello from thread %ld of %d\n", *my_rank, thread_count );

    return nullptr;
}

int main( int argc, char* argv[ ] ){

    thread_count = strtol( argv[ 1 ], nullptr, 10 );

    pthread_t* thread_handles = new pthread_t( thread_count );

    printf( "Hello from the main thread\n" );

    for( long thread = 0; thread < thread_count; ++thread ){

        pthread_create( &thread_handles[ thread ], nullptr, Hello, static_cast<void*>( &thread ) );
    }

    for( long thread = 0; thread < thread_count; ++thread ){

        pthread_join( thread_handles[ thread ], nullptr );
    }

    delete thread_handles;
}