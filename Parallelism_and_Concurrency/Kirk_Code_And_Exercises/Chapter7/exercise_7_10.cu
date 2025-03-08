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

/// 7.10 - Revise the tiled 2D kernel in Fig. 7.12 to perform 3D convolution.

/// ANSWER: 

#define FILTER_RADIUS 1
#define IN_TILE_DIM 16 // I could use extern here instead of a macro, and maybe complicate things, but for now I am avoiding it
#define OUT_TILE_DIM ( ( IN_TILE_DIM ) - ( 2 * ( FILTER_RADIUS ) ) )

__constant__ double filter[ ( 2 * FILTER_RADIUS ) + 1 ][ ( 2 * FILTER_RADIUS ) + 1 ][ ( 2 * FILTER_RADIUS ) + 1 ];

/*__global__ void constM_tiled_convolution_3d( double* input_matrix, double* output_matrix, int height, int width, int depth, int shared_memsize ){

	// Output matrix slice, row, and column
	int slice = ( blockIdx.z * OUT_TILE_DIM ) + threadIdx.z - FILTER_RADIUS;
	int row = ( blockIdx.y * OUT_TILE_DIM ) + threadIdx.y - FILTER_RADIUS;
	int col = ( blockIdx.x * OUT_TILE_DIM ) + threadIdx.x - FILTER_RADIUS;

	//__shared__ double shared_input[ IN_TILE_DIM ][ IN_TILE_DIM ][ IN_TILE_DIM ];
	__shared__ extern double shared_input[];

	if( ( slice >= 0 && slice < depth ) && ( row >= 0 && row < height ) && ( col >= 0 && col < width ) ){

		//shared_input[ threadIdx.z ][ threadIdx.y ][ threadIdx.x ] = input_matrix[ ( slice * depth * depth ) + ( row * height ) + col ];
		shared_input[ (  ) ] = input_matrix[ ( slice * depth * depth ) + ( row * height ) + col ];
	}

	else{

		//shared_input[ threadIdx.z ][ threadIdx.y ][ threadIdx.x ] = 0.0;
		shared_input[  ] = 0.0;
	}

	__syncthreads( );

	int tileSlice = threadIdx.z - FILTER_RADIUS;
	int tileRow = threadIdx.y - FILTER_RADIUS;
	int tileCol = threadIdx.x - FILTER_RADIUS;

	double test{ 0 }, filter_test{ 0 };

	if( ( slice >= 0 && slice < depth ) && ( row >= 0 && row < height ) && ( col >= 0 && col < width ) ){

		if( ( tileSlice >= 0 && tileSlice < OUT_TILE_DIM ) && ( tileRow >= 0 && tileRow < OUT_TILE_DIM ) && ( tileCol >= 0 && tileCol < OUT_TILE_DIM ) ){

			double Pvalue = 0.0;

			for( int fSlice = 0; fSlice < ( ( 2 * FILTER_RADIUS ) + 1 ); ++fSlice ){

				for( int fRow = 0; fRow < ( ( 2 * FILTER_RADIUS ) + 1 ); ++fRow ){

					for( int fCol = 0; fCol < ( ( 2 * FILTER_RADIUS ) + 1 ); ++fCol ){

						test = shared_input[ tileSlice + fSlice ][ tileRow + fRow ][ tileCol + fCol ];
						filter_test = filter[ fSlice ][ fRow ][ fCol ];

						// The problem is here: threads on the block's boundaries are accessing invalid addresses from shared_input
						Pvalue += filter[ fSlice ][ fRow ][ fCol ] * shared_input[ tileSlice + fSlice ][ tileRow + fRow ][ tileCol + fCol ];
					}
				}
			}

			output_matrix[ ( slice * depth * depth ) + ( row * height ) + col ] = Pvalue;
		}
	}
	

}*/

__global__ void constM_tiled_convolution_3d( double* input_matrix, double* output_matrix, int height, int width, int depth, int radius ){

	const int out_tile_dim = blockDim.z - ( 2 * radius );

	// Blocks' dimensions matches input tile + radius BEFORE this function is called
	int in_tile_slice = ( blockDim.z * blockIdx.z ) + threadIdx.z;
	int in_tile_row = ( blockDim.y * blockIdx.y ) + threadIdx.y;
	int in_tile_col = ( blockDim.x * blockIdx.x ) + threadIdx.x;

	// Output matrix is in Global Memory, which means that, if we use multiple blocks, we must use blockDim and blockId to be able to assign threads from different
	// blocks to their corresponding tiles on the output matrix.
	// blockDim - FILTER_RADIUS = output tile dimensions
	// Subtracting FILTER_RADIUS at the end of the expression( after threadIdx ) messes up the following conditional, because output_slice/row/col that should have been greater than 0 are actually equal to it
	int output_slice = ( ( blockDim.z - radius ) * blockIdx.z ) + threadIdx.z - radius;
	int output_row = ( ( blockDim.y - radius ) * blockIdx.y ) + threadIdx.y - radius;
	int output_col = ( ( blockDim.x - radius ) * blockIdx.x ) + threadIdx.x - radius;

	// Shared memory allocation matches block dimensions by definition, i.e., at the call of this function
	__shared__ extern double shared_input[];

	// Loading into shared memory
	if( output_slice >= 0 && output_slice < depth && output_row >= 0 && output_row < height && output_col >= 0 && output_col < width ){

		// By assumption, input and output matrices have the SAME dimensions
		// ATTENTION: shared_input is a TILE, i.e., it is not equal to the entirety of the input matrix.
		//shared_input[ ( in_tile_slice * ( blockDim.z * blockDim.z ) ) + ( in_tile_row * ( blockDim.y - FILTER_RADIUS ) ) + ( blockDim.x - FILTER_RADIUS ) ] = input_matrix[ ( output_slice * depth * depth ) + ( output_row * height ) + output_col ];
		shared_input[ ( threadIdx.z * ( blockDim.z * blockDim.z ) ) + ( threadIdx.y * blockDim.y ) + threadIdx.x ] = input_matrix[ ( output_slice * depth * depth ) + ( output_row * height ) + output_col ];
	}

	else{

		//shared_input[ ( in_tile_slice * ( blockDim.z * blockDim.z ) ) + ( in_tile_row * ( blockDim.y - FILTER_RADIUS ) ) + ( blockDim.x - FILTER_RADIUS ) ] = 0.0;
		shared_input[ ( threadIdx.z * ( blockDim.z * blockDim.z ) ) + ( threadIdx.y * blockDim.y ) + threadIdx.x ] = 0.0;
	}

	__syncthreads( );

	// PROBLEM: this should not be working when ( blockdim - radius ) < input/output_matrix size
		// POTENTIAL ANSWER: since shared_input > input/output matrices dimensions, there is always a subset of threads within valid ranges
		//                   Also, by removing num_threads += radius, things got fixed. I say fuck it.
	/*if( output_slice >= 0 && output_slice < depth && output_row >= 0 && output_row < height && output_col >= 0 && output_col < width ){

		//output_matrix[ ( output_slice * depth * depth ) + ( output_row * height ) + output_col ] = shared_input[ ( in_tile_slice * ( blockDim.z * blockDim.z ) ) + ( in_tile_row * ( blockDim.y - FILTER_RADIUS ) ) + ( blockDim.x - FILTER_RADIUS ) ];
		output_matrix[ ( output_slice * depth * depth ) + ( output_row * height ) + output_col ] = shared_input[ ( threadIdx.z * ( blockDim.z * blockDim.z ) ) + ( threadIdx.y * blockDim.y ) + threadIdx.x ];
	}*/
	
	// NOTE: I still have to explain myself this and why it is necessary to subtract radius from threadId
	int intile_slice = threadIdx.z - radius;
	int intile_row = threadIdx.y - radius;
	int intile_col = threadIdx.x - radius;

	if( output_slice >= 0 && output_slice < depth && output_row >= 0 && output_row < height && output_col >= 0 && output_col < width ){

		if( intile_slice >= 0 && intile_slice < ( blockDim.z - radius ) && intile_row >= 0 && intile_row < ( blockDim.y - radius ) && intile_col >= 0 && intile_col < ( blockDim.x - radius ) ){

			double Pvalue{ 0.0 };

			for( int fSlice = 0; fSlice < ( 2 * radius ) + 1; ++fSlice ){

				for( int fRow = 0; fRow < ( 2 * radius ) + 1; ++fRow ){

					for( int fCol = 0; fCol < ( 2 * radius ) + 1; ++fCol ){

						Pvalue += filter[ fSlice ][ fRow ][ fCol ] * shared_input[ ( ( intile_slice + fSlice ) * ( blockDim.z * blockDim.z ) ) + ( ( intile_row + fRow ) * blockDim.y ) + ( intile_col + fCol ) ];
					}
				}
			}

			output_matrix[ ( output_slice * depth * depth ) + ( output_row * height ) + output_col ] += Pvalue;
		}
	}
}

// NOTE: both input and output matrices have the SAME dimensions
/*void set_up( double*& host_InMatrix, double*& host_filter, double*& host_OutMatrix, int host_matrix_height, int host_matrix_width, int host_matrix_depth ){

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

	//int shared_memsize{ 0 };

	//std::cout << "\nEnter the Shared Memory size: ";
	//std::cin >> shared_memsize;

	//shared_memsize = ( shared_memsize * shared_memsize * shared_memsize ) * sizeof( double );

	//convolution_3D<<<grid, block>>>( dev_InMatrix, dev_filter, dev_OutMatrix, radius, host_matrix_height, host_matrix_width, host_matrix_depth );
	//constM_tiled_convolution_3d<<<grid, block, shared_memsize>>>( dev_InMatrix, dev_OutMatrix, host_matrix_height, host_matrix_width, host_matrix_depth );
	constM_tiled_convolution_3d<<<grid, block>>>( dev_InMatrix, dev_OutMatrix, host_matrix_height, host_matrix_width, host_matrix_depth );

	cudaMemcpy( host_OutMatrix, dev_OutMatrix, matrix_dimensions, cudaMemcpyDeviceToHost );

	cudaFree( dev_InMatrix );
	cudaFree( dev_OutMatrix );
	cudaFree( filter );
}*/

void set_up( double*& host_InMatrix, double*& host_filter, double*& host_OutMatrix, int host_matrix_height, int host_matrix_width, int host_matrix_depth ){

	int radius{ 1 };

	double* dev_InMatrix{ nullptr }, *dev_OutMatrix{ nullptr };

	long int matrix_dimensions = host_matrix_depth * host_matrix_width * host_matrix_height * sizeof( double );
	long int filter_dimensions = ( ( 2 * radius ) + 1 ) * ( ( 2 * radius ) + 1 ) * ( ( 2 * radius ) + 1 ) * sizeof( double );

	cudaMalloc( ( void** ) &dev_InMatrix, matrix_dimensions );
	cudaMalloc( ( void** ) &dev_OutMatrix, matrix_dimensions );

	cudaMemcpy( dev_InMatrix, host_InMatrix, matrix_dimensions, cudaMemcpyHostToDevice );
	//cudaMemcpy( dev_OutMatrix, host_OutMatrix, matrix_dimensions, cudaMemcpyHostToDevice );
	cudaMemcpyToSymbol( filter, host_filter, filter_dimensions ); // Copying data into constant memory

	int num_blocks{ 0 }, num_threads{ 0 };

	std::cout << "Enter the amount of blocks and threads: ";
	std::cin >> num_blocks >> num_threads;

	num_threads += radius;

	dim3 grid( num_blocks, 1, 1 );
	dim3 block( num_threads, num_threads, num_threads );

	int shared_memsize = ( num_threads * num_threads * num_threads ) * sizeof( double );

	//std::cout << "\nEnter the Shared Memory size: ";
	//std::cin >> shared_memsize;

	//shared_memsize = ( shared_memsize * shared_memsize * shared_memsize ) * sizeof( double );

	//convolution_3D<<<grid, block>>>( dev_InMatrix, dev_filter, dev_OutMatrix, radius, host_matrix_height, host_matrix_width, host_matrix_depth );
	//constM_tiled_convolution_3d<<<grid, block, shared_memsize>>>( dev_InMatrix, dev_OutMatrix, host_matrix_height, host_matrix_width, host_matrix_depth );
	constM_tiled_convolution_3d<<<grid, block, shared_memsize>>>( dev_InMatrix, dev_OutMatrix, host_matrix_height, host_matrix_width, host_matrix_depth, radius );

	cudaMemcpy( host_OutMatrix, dev_OutMatrix, matrix_dimensions, cudaMemcpyDeviceToHost );

	cudaFree( dev_InMatrix );
	cudaFree( dev_OutMatrix );
	cudaFree( filter );
}

/// PROBLEM: for the program to run correctly, the block size must be GREATER THAN the input matrix size

int main( ){

	int matrix_dim{ 0 }, filter_dim{ 0 };

	std::cout << "Enter the dimensions of the matrix: ";
	std::cin >> matrix_dim;

	std::cout << "\nEnter the dimensions of the filter matrix: ";
	std::cin >> filter_dim;

	std::cout << "\n";

	double*** matrix = generate_matrix( matrix_dim );
	double*** filter = generate_matrix( filter_dim );

	double* flat_matrix = flatten_matrix( matrix, matrix_dim );
	double* flat_filter = flatten_matrix( filter, filter_dim );

	double* output = new double[ matrix_dim * matrix_dim * matrix_dim * sizeof( double ) ];

	print_matrix( matrix, matrix_dim );

	std::cout << "\n";

	//print_matrix( flat_matrix, 5 );

	//std::cout << "\n";

	print_matrix( flat_filter, filter_dim );

	std::cout << "\n";

	set_up( flat_matrix, flat_filter, output, matrix_dim, matrix_dim, matrix_dim );

	std::cout << "\n";

	print_matrix( output, matrix_dim );

	delete[ ] matrix, filter, output;
	delete[ ] flat_matrix, flat_filter;
}