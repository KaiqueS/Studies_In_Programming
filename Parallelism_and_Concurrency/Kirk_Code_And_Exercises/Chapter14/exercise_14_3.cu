
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include "\Projetos\Studies_In_Programming\Parallelism_and_Concurrency\Kirk_Code_And_Exercises\Chapter14\CUDA_Sparse_Matrixes\Sparse_Matrixes.cpp"

#include <stdio.h>
#include <iostream>
#include <random>

int** generate_matrix( int row, int col ){

	int** matrix = new int*[ row ];

	std::random_device dev;
	std::uniform_int_distribution<int> dist( -( row * col ), ( row * col ) );
	std::mt19937_64 rng( dev( ) );

	for( auto i = 0; i < row; ++i ){

		matrix[ i ] = new int[ col ];

		for( auto j = 0; j < col; ++j ){

			matrix[ i ][ j ] = dist( rng );
		}
	}

	return matrix;
}

void print( int**& matrix, int row, int col ){

	for( auto i = 0; i < row; ++i ){

		for( auto j = 0; j < col; ++j ){

			printf( "%d ", matrix[ i ][ j ] );
		}

		printf( "\n" );
	}

	printf( "\n" );
}

void print( const std::vector<int>& matrix ){

	for( auto i = 0; i < matrix.size( ); ++i ){

		printf( "%d ", matrix[ i ] );
	}

	printf( "\n" );
}

/// PROBLEM: 3. Implement the code to convert from COO to CSR using fundamental parallel computing primitives, including histogram and prefix sum.

__global__ void COO_CSR_Kernel( COO* input, CSR* output ){


}

void kernel_setup( COO*& host_input, CSR*& host_output ){

	COO* dev_input{ };
	CSR* dev_output{ };

	int coo_size = sizeof( COO* );
	int csr_size = sizeof( CSR* );

	cudaMalloc( ( void** ) &dev_input, coo_size );
	cudaMalloc( ( void** ) &dev_output, csr_size );

	cudaMemcpy( dev_input, host_input, coo_size, cudaMemcpyHostToDevice );
	
	unsigned int gridsize{ 0 }, blocksize{ 0 };

	std::cout << "Enter the number of blocks: ";
	std::cin >> gridsize;

	std::cout << "\nEnter the number of threads: ";
	std::cin >> blocksize;

	dim3 blocks{ gridsize };
	dim3 threads{ blocksize };

	COO_CSR_Kernel<<<blocks, threads>>>( dev_input, dev_output );

	cudaMemcpy( host_output, dev_output, csr_size, cudaMemcpyDeviceToHost );

	cudaFree( dev_input );
	cudaFree( dev_output );
}

int main( ){

	int rowsize{ 0 }, colsize{ 0 };

	std::cout << "Enter the dimensions of the matrix: ";
	std::cin >> rowsize >> colsize;

	int** matrix = generate_matrix( rowsize, colsize );

	print( matrix, rowsize, colsize );

	COO coo_mtx{ };
	coo_mtx.build_COO( matrix, rowsize, colsize );

	print( coo_mtx.get_value( ) );
}