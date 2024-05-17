#include <iostream>
#include <mpi.h>
#include <functional>

double quadratic( double value ){

    return value * value;
}

double Trapezoid( double left_end, double right_end,  int trap_count, double base_len, std::function<double( double )> f ){

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

int main( ){

    int my_rank{ }, comm_sz{ };

    MPI_Init( nullptr, nullptr );
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &comm_sz );

    MPI_Finalize( );
}