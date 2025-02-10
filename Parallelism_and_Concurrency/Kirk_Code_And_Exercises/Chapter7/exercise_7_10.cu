
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>
#include <iostream>

double***& generate_matrix( int size ){

	std::random_device dev;
	std::uniform_real_distribution<double> dist( -( size * size ), ( size * size ) );
	std::mt19937_64 rng( dev( ) );

	double*** matrix = new double**[ size ];

	for( auto i = 0; i < size; ++i ){

		matrix[ i ] = new double*[ size ];

		for( auto j = 0; j < size; ++j ){

			matrix[ i ][ j ] = new double[ size ];

			for( int k = 0; k < size; ++k ){

				matrix[ i ][ j ][ k ] = dist( rng );
			}
		}
	}

	return matrix;
}

double*& flatten_matrix( double***& matrix, int size ){

	double* flat = new double[ size * size * size ];

	for( auto i = 0; i < size; ++i ){

		for( auto j = 0; j < size; ++j ){

			for( auto k = 0; k < size; ++k ){

				flat[ ( i * size * size ) + ( j * size ) + k ] = matrix[ i ][ j ][ k ];
			}
		}
	}

	return flat;
}

double**& generate_filter( int size ){

	std::random_device dev;
	std::uniform_real_distribution<double> dist( -( size * size ), ( size * size ) );
	std::mt19937_64 rng( dev( ) );

	double** filter = new double*[ size ];

	for( auto i = 0; i < size; ++i ){

		filter[ i ] = new double[ size ];

		for( auto j = 0; j < size; ++j ){

			filter[ i ][ j ] = dist( rng );
		}
	}

	return filter;
}

double*& flatten_filter( double**& filter, int size ){

	double* out = new double[ size * size ];

	for( auto i = 0; i < size; ++i ){

		for( auto j = 0; j < size; ++j ){

			out[ ( i * size ) + j ] = filter[ i ][ j ];
		}
	}

	return out;
}

void print_matrix( double*** matrix, int size ){

	for( auto i = 0; i < size; ++i ){

		for( auto j = 0; j < size; ++j ){

			for( auto k = 0; k < size; ++k ){

				printf( "%f ", matrix[ i ][ j ][ k ] );
			}

			printf( "\n" );
		}

		printf( "\n" );
	}
}

void print_matrix( double* matrix, int size ){

	for( auto i = 0; i < size; ++i ){

		for( auto j = 0; j < size; ++j ){

			for( auto k = 0; k < size; ++k ){

				printf( "%f ", matrix[ ( i * size * size ) + ( j * size ) + k ] );
			}

			printf( "\n" );
		}

		printf( "\n" );
	}
}

void print_filter( double* filter, int size ){

	for( auto i = 0; i < size; ++i ){

		for( auto j = 0; j < size; ++j ){

			printf( "%f ", filter[ ( i * size ) + j ] );
		}

		printf( "\n" );
	}
}

/// 10. Revise the tiled 2D kernel in Fig. 7.12 to perform 3D convolution.

/// ANSWER: 

#define FILTER_RADIUS 1
#define IN_TILE_DIM 32 // I could use extern here and maybe complicate things, but for now I am avoiding it
#define OUT_TILE_DIM ( ( IN_TILE_DIM ) - ( 2 * ( FILTER_RADIUS ) ) )

__constant__ double filter[ ( 2 * FILTER_RADIUS ) + 1 ][ ( 2 * FILTER_RADIUS ) + 1 ][ ( 2 * FILTER_RADIUS ) + 1 ];

// NOTE: both input and output matrices have the SAME dimensions
void set_up( double*& host_InMatrix, double*& host_filter, double*& host_OutMatrix, int host_matrix_height, int host_matrix_width, int host_matrix_depth ){

	double* dev_InMatrix{ nullptr }, *dev_OutMatrix{ nullptr };

	long int matrix_dimensions = host_matrix_depth * host_matrix_width * host_matrix_height * sizeof( double );
	long int filter_dimensions = ( ( 2 * FILTER_RADIUS ) + 1 ) * ( ( 2 * FILTER_RADIUS ) + 1 ) * ( ( 2 * FILTER_RADIUS ) + 1 ) * sizeof( double );

	cudaMalloc( ( void** ) &dev_InMatrix, matrix_dimensions );
	cudaMalloc( ( void** ) &dev_OutMatrix, matrix_dimensions );

	cudaMemcpy( dev_InMatrix, host_InMatrix, matrix_dimensions, cudaMemcpyHostToDevice );
	cudaMemcpy( dev_OutMatrix, host_OutMatrix, matrix_dimensions, cudaMemcpyHostToDevice );
	cudaMemcpyToSymbol( filter, host_filter, filter_dimensions ); // Copying data into constant memory

	int num_blocks{ 0 }, num_threads{ 0 };

	std::cout << "Enter the amount of blocks and threads: ";
	std::cin >> num_blocks >> num_threads;

	dim3 grid( num_blocks,1, 1 );
	dim3 block( num_threads, num_threads, num_threads );

	//	convolution_3D<<<grid, block>>>( dev_InMatrix, dev_filter, dev_OutMatrix, radius, host_matrix_height, host_matrix_width, host_matrix_depth );
	convolution_3D<<<grid, block>>>( dev_InMatrix, dev_OutMatrix, host_matrix_height, host_matrix_width, host_matrix_depth );

	cudaMemcpy( host_OutMatrix, dev_OutMatrix, matrix_dimensions, cudaMemcpyDeviceToHost );

	cudaFree( dev_InMatrix );
	cudaFree( dev_OutMatrix );
}

int main( ){

	double*** matrix = generate_matrix( 5 );
	double*** filter = generate_matrix( 3 );

	double* flat_matrix = flatten_matrix( matrix, 5 );
	double* flat_filter = flatten_matrix( filter, 3 );

	double* output = new double[ 5 * 5 * 5 * sizeof( double ) ];

	print_matrix( matrix, 5 );

	std::cout << "\n";

	//print_matrix( flat_matrix, 5 );

	//std::cout << "\n";

	print_matrix( flat_filter, 3 );

	std::cout << "\n";

	set_up( flat_matrix, flat_filter, output, 5, 5, 5 );

	std::cout << "\n";

	print_matrix( output, 5 );

	delete[ ] matrix, filter, output;
	delete[ ] flat_matrix, flat_filter;
}