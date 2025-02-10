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

/// 9. Revise the 2D kernel in Fig. 7.9 to perform 3D convolution.

/// ANSWER: 

#define FILTER_RADIUS 1

__constant__ double filter[ ( 2 * FILTER_RADIUS ) + 1 ][ ( 2 * FILTER_RADIUS ) + 1 ][ ( 2 * FILTER_RADIUS ) + 1 ];

__global__ void convolution_3D( double* input_matrix, double* output_matrix, int matrix_width, int matrix_height, int matrix_depth  ){
	
	// DESCRIPTION: take the radius R, and pick an element E from the input matrix M, where the indexes of E are x, y, z. Then, all elements within the 3D halo of E in M that share the
	//							same x, y coordinates but differ on z will be multiplied by the same element from the filter. I.e., to get a better visualization, imagine that we are building a 3D ma-
	//							trix with ( 2R + 1 ) slices, where all slices are equal to the filter.

	int outCol = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	int outRow = ( blockIdx.y * blockDim.y ) + threadIdx.y;
	int outSlice = ( blockIdx.z * blockDim.z ) + threadIdx.z;

	double Pvalue = 0.0f;

	// double filter_element{ 0 }, matrix_element{ 0 }; DEBUG ONLY

	for( auto fSlice = 0; fSlice < ( ( 2 * FILTER_RADIUS ) + 1 ); ++fSlice ){

		for( auto fRow = 0; fRow < ( ( 2 * FILTER_RADIUS ) + 1 ); ++fRow ){

			for( auto fCol = 0; fCol < ( ( 2 * FILTER_RADIUS ) + 1 ); ++fCol ){

				int inRow = outRow - FILTER_RADIUS + fRow;
				int inCol = outCol - FILTER_RADIUS + fCol;
				int inSlice = outSlice - FILTER_RADIUS + fSlice;

				if( inRow >= 0 && inRow < matrix_height &&
					 inCol >= 0 && inCol < matrix_width &&
					 inSlice >= 0 && inSlice < matrix_depth ){

					// filter_element = filter[ ( fRow * ( ( 2 * radius ) + 1 ) ) + fCol ]; DEBUG ONLY
					// matrix_element = input_matrix[ ( inSlice * matrix_depth * matrix_depth ) + ( inRow * matrix_width ) + inCol ]; DEBUG ONLY

					// Here, since our filter is 2D, there is NO need to iterate over its slices, since we are using only one slice repeatedly.
					Pvalue += filter[ fSlice ][ fRow ][ fCol ] * input_matrix[ ( inSlice * matrix_depth * matrix_depth ) + ( inRow * matrix_width ) + inCol ];
				}
			}
		}
	}

	output_matrix[ ( outSlice * matrix_depth * matrix_depth ) + ( outRow * matrix_width ) + outCol ] = Pvalue;
}

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