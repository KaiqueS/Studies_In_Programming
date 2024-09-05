#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <random>

float*& simple_flattening( float**& matrix, int size ){

	float* flat = new float[ size * size ];

	for( auto i = 0; i < size; ++i ){

		for( auto j = 0; j < size; ++j ){

			flat[ ( i * size ) + j ] = matrix[ i ][ j ];
		}
	}

	return flat;
}

void fill_matrix( float**& matrix, int size ){

	std::random_device dev;
	std::uniform_real_distribution<float> dist( -( size * size ), ( size * size ) );
	std::mt19937_64 rng( dev( ) );

	matrix = new float*[ size ];

	for( auto i = 0; i < size; ++i ){

		matrix[ i ] = new float[ size ];

		for( auto j = 0; j < size; ++j ){

			matrix[ i ][ j ] = dist( rng );
		}
	}
}

void fill_vector( float*& vetor, int size ){

	std::random_device dev;
	std::uniform_real_distribution<float> dist( -( size * size ), ( size * size ) );
	std::mt19937_64 rng( dev( ) );

	vetor = new float[ size ];

	for( auto i = 0; i < size; ++i ){

		vetor[ i ] = dist( rng );
	}
}

void print_matrix( float**& matrix, int size ){

	for( auto i = 0; i < size; ++i ){

		for( auto j = 0; j < size; ++j ){

			printf( "%lf ", matrix[ i ][ j ] );

			if( j == size - 1 ){

				printf( "\n" );
			}
		}
	}
}

void print_flat_matrix( float*& matrix, int row, int col ){

	for( auto i = 0; i < row; ++i ){

		for( auto j = 0; j < col; ++j ){

			printf( "%lf ", matrix[ ( i * row ) + j ] );
		}

		printf( "\n" );
	}
}

void print_vector( float*& vetor, int size ){

	for( auto i = 0; i < size; ++i ){

		printf( "%lf ", vetor[ i ] );
	}
}

/// NOTE: I have to come up with a way to make blockDim a function of matrix size

/// 2. A matrix–vector multiplication takes an input matrix B and a vector C and produces one output vector A. Each element of the output vector A is the dot
/// product of one row of the input matrix B and C, i.e., A[ i ] = ∑j B[ i ][ j ] + C[ j ]. For simplicity, we will only handle square matrices whose elements
/// are single-precision floating-point numbers. Write a matrix–vector multiplication kernel and a host stub function that can be called with four parameters:
/// pointer-to-the-output matrix, pointer-to-the-input matrix, pointer-to-the-input vector, and the number of elements in each dimension. Use one thread to
/// calculate an output vector element.

__global__ void matrixMultKernel( float* output, float* left, float* right, int dim ){

	int col = ( blockIdx.x * blockDim.x ) + threadIdx.x;

	if( col < dim ){

		for( auto i = 0; i < dim; ++i ){

			output[ col ] += left[ ( col * dim ) + i ] * right[ i ];
		}
	}
}

void set_up( float*& host_out, float**& host_matrix, float*& host_vetor, int dim ){

	float* dev_out{ nullptr }, *dev_matrix{ nullptr }, *dev_vetor{ nullptr };
	float* flat_matrix{ simple_flattening( host_matrix, dim ) };

	int matrix_size = ( dim * dim ) * sizeof( float );
	int vetor_size = dim * sizeof( float );

	cudaMalloc( ( void** ) &dev_matrix, matrix_size );
	cudaMalloc( ( void** ) &dev_vetor, vetor_size );
	cudaMalloc( ( void** ) &dev_out, vetor_size );

	cudaMemcpy( dev_matrix, flat_matrix, matrix_size, cudaMemcpyHostToDevice );
	cudaMemcpy( dev_vetor, host_vetor, vetor_size, cudaMemcpyHostToDevice );

	int blocks{ 0 }, threads{ 0 };

	printf( "Enter Grid and Block sizes:\n" );
	std::cin >> blocks >> threads;

	dim3 grid( blocks, 1, 1 );
	dim3 block( threads, 1, 1 );

	matrixMultKernel<<<grid, block>>>( dev_out, dev_matrix, dev_vetor, dim );

	host_out = new float[ dim ];

	cudaMemcpy( host_out, dev_out, vetor_size, cudaMemcpyDeviceToHost );

	cudaFree( dev_out );
	cudaFree( dev_matrix );
	cudaFree( dev_vetor );
}

int main( ){

	float** leftM{ nullptr };
	float* output{ nullptr }, *vetor{ nullptr };

	int dim{ 0 };

	printf( "Enter the matrix dimensions:\n" );
	std::cin >> dim;

	fill_matrix( leftM, dim );
	fill_vector( vetor, dim );

	printf( "\n" );

	print_matrix( leftM, dim );

	printf( "\n" );

	print_vector( vetor, dim );

	printf( "\n" );

	set_up( output, leftM, vetor, dim );

	print_vector( output, dim );

	printf( "\n" );

	delete[ ] output, vetor, leftM;
}
