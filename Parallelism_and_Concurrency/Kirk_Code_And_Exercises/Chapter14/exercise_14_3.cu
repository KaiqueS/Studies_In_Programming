
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "\Projetos\Studies_In_Programming\Parallelism_and_Concurrency\Kirk_Code_And_Exercises\Chapter14\CUDA_Sparse_Matrixes\Sparse_Arrays.cpp"

#include <stdio.h>
#include <iostream>
#include <random>
#include <math.h>

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

void print( const int* matrix, int size ){

	for( auto i = 0; i < size; ++i ){

		printf( "%d ", matrix[ i ] );
	}

	printf( "\n" );
}

/// PROBLEM: 3. Implement the code to convert from COO to CSR using fundamental parallel computing primitives, including histogram and prefix sum.

// IDEA:

	// HISTOGRAM: count the amount of elements from each ROW in COO MATRIX. Use both rowIdx and colIDx to avoid double counting elements

	// PREFIX SUM: scan the histogram, so a to calculate the offset from one row to another

	// Also: csr.colIdx = coo.colIdx and csr.value = coo.value. I.e., we just need to correctly build csr.rowPtrs from coo.rowIdx
__global__ void COO_CSR_Kernel( COO* input, CSR* output ){

	int globalIndex = ( blockDim.x * blockIdx.x ) + threadIdx.x;
	
	int* rows{ nullptr }; // NOTE: rows might not be in global memory! Careful, here.

	int maxRows{ 0 };

	// HISTOGRAM: first, divide COO::rowIdx.size() by gridDim.x, so that each block is responsible for equally-sized subarrays of COO::rowIdx
	//			  second, if the subarray size is greater than blockDim, then, divide subarray.size by blockDim, so that each thread within the block is responsible for the same amount of elements from the subarray
	//			  third, for a vector rows from 0 to std::max( COO::rowIdx ), find the corresponding element in rows and perform an atomicAdd	
	if( globalIndex == 0 ){

		for( auto i = 0; i < input -> get_size( ); ++i ){

			if( input -> get_rowIdx( )[ i ] > maxRows ){

				maxRows = input -> get_rowIdx( )[ i ];
			}
		}

		rows = new int[ maxRows ]{ 0 };

		for( auto i = 0; i < maxRows; ++i ){

			rows[ i ] = i;
		}
	}

	int coo_size = input -> get_size( );

	// If the amount of elements in coo_rowIdx is greater than the amount of threads, assign each block to a portion of coo_rowIdx
	if( coo_size > static_cast<long long int>( gridDim.x * blockDim.x ) ){

		// If the portion assigned to each block is greater than the block, assign each thread to a subportion( with at least one element assigned to each thread )
		if( ( coo_size / gridDim.x ) > blockDim.x ){

			int elements = static_cast<int>( std::ceil( static_cast<double>( coo_size ) / static_cast<double>( gridDim.x ) ) );

			// Thread coarsening
			for( auto i = 0; i < elements; ++i ){

				// Boundary checking
				if( ( globalIndex + i ) < coo_size ){

					for( auto row_idx = 0; row_idx < maxRows; ++row_idx ){

						// Finding the corresponding row
						if( rows[ row_idx ] == input -> get_rowIdx( )[ globalIndex + i ] ){

							atomicAdd( &rows[ row_idx ], 1 );
						}

						// NOTE: these continues MIGHT be troublesome.
						else{

							continue;
						}
					}
				}

				else{

					continue;
				}
			}
		}

		// Else, each thread handles at most one element
		else{

			for( auto row_idx = 0; row_idx < maxRows; ++row_idx ){

				// Finding the corresponding row
				if( rows[ row_idx ] == input -> get_rowIdx( )[ globalIndex ] ){

					atomicAdd( &rows[ row_idx ], 1 );
				}

				else{

					continue;
				}
			}
		}
	}

	else{

		for( auto row_idx = 0; row_idx < maxRows; ++row_idx ){

			// Finding the corresponding row
			if( rows[ row_idx ] == input -> get_rowIdx( )[ globalIndex ] ){

				atomicAdd( &rows[ row_idx ], 1 );
			}

			else{

				continue;
			}
		}
	}

	/// PREFIX SUM: just scan the vector rows
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

	COO* coo_mtx{ };
	coo_mtx -> build_COO( matrix, rowsize, colsize );

	CSR* csr_mtx{ };
	csr_mtx -> build_CSR( matrix, rowsize, colsize );

	kernel_setup( coo_mtx, csr_mtx );

	//print( coo_mtx.get_value( ), coo_mtx.get_size( ) );
}