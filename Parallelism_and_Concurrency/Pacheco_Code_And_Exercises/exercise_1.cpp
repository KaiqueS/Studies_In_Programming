#include <iostream>
#include <vector>
#include <mpi.h>
#include <string.h>

// 3.1 What happens in the greetings program if, instead of strlen(greeting) + 1,
// we use strlen(greeting) for the length of the message being sent by pro-
// cesses 1, 2, . . ., comm_sz−1? What happens if we use MAX_STRING instead of
// strlen( greeting) + 1? Can you explain these results?

/// ANSWER: nothing happens if we use the actual length of the string, instead of length + 1.
///         But, keep in mind - I'm using std::string, instead of std::char here. The method
///         .size() might actually be strlen(string) + 1. Now, if I use a manually set size,
///         well..., unless the set size at the MPI_RECV call is at most the actual string size,
///         then the program crashes from a segmentation fault, and the message gets truncated,
///         because the program will try to read from outside of the buffer reserved for the
///         message.
/// NOTE:   there is no need to declared the variables outside of the process-branching if-elses.
///         I.e., we can let entities be process-local, and the communication will still be
///         successful.

int main( ){

    int my_rank{ }, comm_sz{ };

    MPI_Init( NULL, NULL );
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &comm_sz );

    std::string greetings{ "Hello from process " };
    greetings.append( std::to_string( my_rank ) );

    int greet_size{ greetings.size( ) };

    if( my_rank != 0 ){

        MPI_Send( &greet_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD );
        MPI_Send( greetings.data( ), greetings.size( ), MPI_CHAR, 0, 0, MPI_COMM_WORLD );
    }

    else{

        std::cout << greetings << "\n";

        for( int i = 1; i < comm_sz; ++i ){

            MPI_Recv( &greet_size, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
            MPI_Recv( greetings.data( ), greet_size, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );

            std::cout << greetings << "\n";
        }
    }

    MPI_Finalize( );

    return 0;
}