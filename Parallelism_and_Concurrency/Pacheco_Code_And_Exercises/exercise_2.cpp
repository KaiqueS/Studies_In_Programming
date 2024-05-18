#include <iostream>
#include <mpi.h>
#include <functional>

class mpi_input{

public:

    mpi_input( ){ }
    ~mpi_input( ){

        MPI_Type_free( &input_mpi_t );
    }

    MPI_Datatype& datatype( ){ return input_mpi_t; }

    void Build_mpi_type( double* a_p, double* b_p, int* n_p ){

        int array_of_blocklengths[ 3 ] = { 1, 1, 1 };

        MPI_Datatype array_of_types[ 3 ] = { MPI_DOUBLE, MPI_DOUBLE, MPI_INT };

        MPI_Aint a_addr, b_addr, n_addr;
        MPI_Aint array_of_displacements[ 3 ] = { 0 };

        MPI_Get_address( a_p, &a_addr );
        MPI_Get_address( b_p, &b_addr );
        MPI_Get_address( n_p, &n_addr );

        array_of_displacements[ 1 ] = b_addr - a_addr;
        array_of_displacements[ 2 ] = n_addr - a_addr;

        MPI_Type_create_struct( 3, array_of_blocklengths, array_of_displacements, array_of_types, &input_mpi_t );
        MPI_Type_commit( &input_mpi_t );
    }

private:

    MPI_Datatype input_mpi_t;
};

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

void Get_Input( int my_rank, int comm_sz, double* a_p, double* b_p, int* n_p ){

    mpi_input input{ };
    input.Build_mpi_type( a_p, b_p, n_p );

    if( my_rank == 0 ){

        printf( "Enter a: " );
        scanf( "%lf", a_p );

        printf( "\nEnter b: " );
        scanf( "%lf", b_p );

        printf( "\nEnter n: " );
        scanf( "%lf", n_p );
    }

    MPI_Bcast( a_p, 1, input.datatype( ), 0, MPI_COMM_WORLD );
}

int main( ){

    int my_rank{ }, comm_sz{ };

    MPI_Init( nullptr, nullptr );
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &comm_sz );

    int trapezoid_count{ }, local_trapezoid{ };
    double left_point{ }, right_point{ }, local_left{ }, local_right{ }, height{ };
    double total_estimate{ }, local_estimate{ };

    Get_Input( my_rank, comm_sz, &left_point, &right_point, &trapezoid_count );

    height = ( right_point - left_point ) / trapezoid_count;
    local_trapezoid = trapezoid_count / comm_sz;

    local_left = left_point + my_rank * local_trapezoid * height;
    local_right = local_left + local_trapezoid * height;
    
    local_estimate = Trapezoid( local_left, local_right, local_trapezoid, height, quadratic );

    if( my_rank != 0 ){

        
    }

    MPI_Finalize( );
}