#include <iostream>
#include <mpi.h>

 int main( ){

    int my_rank{ 0 }, comm_sz{ 0 };

    MPI_Init( nullptr, nullptr );
    MPI_Comm_rank( MPI_COMM_WORLD,  &my_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &comm_sz );

    MPI_Finalize( );
 }