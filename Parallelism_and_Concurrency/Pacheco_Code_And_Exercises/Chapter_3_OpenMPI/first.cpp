#include <iostream>
#include <string>
#include <stdio.h>
#include <functional>
#include <mpi.h>

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

    int dest{ };

    if( my_rank == 0 ){

        printf( "Enter a, b, and n\n" );

        scanf( "%lf %lf %d", a_p, b_p, n_p );

        for( dest = 1; dest < comm_sz; ++dest ){

            MPI_Send( a_p, 1, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD );
            MPI_Send( b_p, 1, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD );
            MPI_Send( n_p, 1, MPI_INT, dest, 0, MPI_COMM_WORLD );
        }
    }

    else{

        MPI_Recv( a_p, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        MPI_Recv( b_p, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        MPI_Recv( n_p, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
    }
}

int main( void ){

    int my_rank{ }, comm_sz{ }, n{ 1024 }, local_n{ };

    double a{ 0.0 }, b{ 3.0 }, h{ }, local_a{ }, local_b{ };
    double local_int{ }, total_int{ };

    int source{ };

    MPI_Init( NULL, NULL );
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &comm_sz );

    Get_input( my_rank, comm_sz, &a, &b, &n );

    h = ( b - a ) / n;
    
    local_n = n / comm_sz;
    local_a = a + my_rank * local_n * h;
    local_b = local_a + local_n * h;
    local_int = Trap( local_a, local_b, local_n, h, quadratic );

    if( my_rank != 0 ){

        MPI_Send( &local_int, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD );
    }

    else{

        total_int = local_int;

        for( source = 1; source < comm_sz; ++source ){

            MPI_Recv( &local_int, 1, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );

            total_int += local_int;
        }
    }

    if( my_rank == 0 ){

        printf( "With n = %d trapezoids, our estimate\n", n );
        printf( "of the integral from %f to %f = %.15e\n", a, b, total_int );
    }

    MPI_Finalize( );

    return 0;
}