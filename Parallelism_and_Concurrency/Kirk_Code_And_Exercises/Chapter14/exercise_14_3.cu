
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
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

	for( auto i = 0; i < input -> get_size( ); ++i ){

		if( input -> get_rowIdx( )[ i ] > maxRows ){

			maxRows = input -> get_rowIdx( )[ i ];
		}
	}

	// HISTOGRAM: first, divide COO::rowIdx.size() by gridDim.x, so that each block is responsible for equally-sized subarrays of COO::rowIdx
	//			  second, if the subarray size is greater than blockDim, then, divide subarray.size by blockDim, so that each thread within the block is responsible for the same amount of elements from the subarray
	//			  third, for a vector rows from 0 to std::max( COO::rowIdx ), find the corresponding element in rows and perform an atomicAdd	
	if( globalIndex == 0 ){

		//cudaMalloc( ( void** ) &rows, maxRows * sizeof( int ) );

		rows = new int[ maxRows ]{ 0 };

		for( auto i = 0; i < maxRows; ++i ){

			rows[ i ] = i;
			//memcpy( &( rows[ i ] ), &i, sizeof( int ) );
		}
	}

	// NOTE: I need cross-block and cross-warp sync here, else, threads with globalIndex != 0 just pass through loc 85 BEFORE rows is properly initialized.

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

	if( globalIndex == 0 ){

		output -> get_colIdx( ) = input -> get_colIdx( );
		output -> get_value( ) = input -> get_value( );
		
		for( auto i = 0; i < maxRows; ++i ){

			output -> get_rowPtrs( )[ i ] = rows[ i ];
		}

		free( rows );
	}
}

void kernel_setup( COO*& host_input, CSR*& host_output ){

	COO* dev_input{ new COO };
	CSR* dev_output{ new CSR };

	//dev_input -> rowIdx = new int[ host_input -> get_size( ) ];

	int coo_size = sizeof( COO );
	int csr_size = sizeof( CSR );

	cudaMalloc( ( void** ) &dev_input, coo_size );
	cudaMalloc( ( void** ) &dev_output, csr_size );

	cudaMemcpy( dev_input, host_input, coo_size, cudaMemcpyHostToDevice );

	/// NOTE - START

	// The following code is required for the members of COO/CSR to be correctly copied from host data. Why?
	// Because, since some members of these classes are actually pointers to, instead of actual types, the
	// cudaMemcpy call at loc 188 is not enough to copy the members from host_input onto the the members of
	// dev_input. -
	// Why? My guess - it is a copy by value, not by reference. The copying does not pass the addresses from the
	// members of host_input, because these addresses are in host memory address space, which is different from
	// device address space. Thus, they must be copied by value, and, even if it does copy the address from the
	// host members, it is an invalid address wrt the device address space. -
	// But why cannot we copy straight from host_input onto dev_input?
	// Because, upon allocation of dev_input, we cannot allocate space specifically for dev_input pointer members.
	// I.e., we cannot make another cudaMalloc passing dev_input -> rowIdx/colIdx/value, since it would invalidate,
	// on the device, the address allocated for dev_input on loc 185. Thus, we must create, on the host code,
	// pointers to the same type of the members from COO/CSR, allocate space for these host pointers, copy
	// data from host to these host pointers, then, make the members pointers of dev_input, on the device, point
	// to the space allocated, on the device, for the host pointers. - 
	int* rowIdx{ nullptr }, *colIdx{ nullptr }, *value{ nullptr };
	cudaMalloc( ( void** ) &rowIdx, host_input -> get_size( ) * sizeof( int ) );
	cudaMalloc( ( void** ) &colIdx, host_input -> get_size( ) * sizeof( int ) );
	cudaMalloc( ( void** ) &value, host_input -> get_size( ) * sizeof( int ) );

	cudaMemcpy( rowIdx, host_input -> get_rowIdx( ), host_input -> get_size( ) * sizeof( int ), cudaMemcpyHostToDevice );
	cudaMemcpy( colIdx, host_input -> get_colIdx( ), host_input -> get_size( ) * sizeof( int ), cudaMemcpyHostToDevice );
	cudaMemcpy( value, host_input -> get_value( ), host_input -> get_size( ) * sizeof( int ), cudaMemcpyHostToDevice );

	cudaMemcpy( &( dev_input -> get_rowIdx( ) ), &rowIdx, sizeof( int* ), cudaMemcpyHostToDevice );
	cudaMemcpy( &( dev_input -> get_colIdx( ) ), &colIdx, sizeof( int* ), cudaMemcpyHostToDevice );
	cudaMemcpy( &( dev_input -> get_value( ) ), &value, sizeof( int* ), cudaMemcpyHostToDevice );
	/// NOTE - END

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
	cudaFree( rowIdx );
}

int main( ){

	int rowsize{ 0 }, colsize{ 0 };

	std::cout << "Enter the dimensions of the matrix: ";
	std::cin >> rowsize >> colsize;

	int** matrix = generate_matrix( rowsize, colsize );

	print( matrix, rowsize, colsize );

	COO* coo_mtx{ new COO };
	coo_mtx -> build_COO( matrix, rowsize, colsize );

	CSR* csr_mtx{ new CSR };
	//csr_mtx -> build_CSR( matrix, rowsize, colsize );

	kernel_setup( coo_mtx, csr_mtx );

	print( csr_mtx -> get_value( ), csr_mtx -> get_size( ) );
}