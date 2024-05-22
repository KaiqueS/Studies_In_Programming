#include <iostream>
#include <mpi.h>
#include <string>
#include <sstream>

// 3.4 Modify the program that just prints a line of output from each process
// (mpi_output.c) so that the output is printed in process rank order: process 0’s
// output first, then process 1’s, and so on.

/// ANSWER: check code

 int main( ){

   int my_rank{ 0 }, comm_sz{ 0 };

   MPI_Init( nullptr, nullptr );
   MPI_Comm_rank( MPI_COMM_WORLD,  &my_rank );
   MPI_Comm_size( MPI_COMM_WORLD, &comm_sz );

   if( my_rank != 0 ){

      std::string message{ ( std::ostringstream( ) << "Process " << my_rank << " of " <<  comm_sz << " > Does anyone have a toothpick?\n" ).str( ) };

      MPI_Send( message.data( ), message.size( ), MPI_CHAR, 0, 0, MPI_COMM_WORLD );
   }

   else{

      std::string message{ ( std::ostringstream( ) << "Process " << my_rank << " of " <<  comm_sz << " > Does anyone have a toothpick?\n" ).str( ) };

      printf( message.data( ) );

      for( int process = 1; process < comm_sz; ++process ){

         MPI_Recv( message.data( ), message.size( ), MPI_CHAR, process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );

         printf( message.data( ) );
      }
   }

   MPI_Finalize( );
 }