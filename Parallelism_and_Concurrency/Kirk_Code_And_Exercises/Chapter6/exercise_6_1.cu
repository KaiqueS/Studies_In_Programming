#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <random>
#include <iostream>

double**& generate_matrix( int size ){

	std::random_device dev;
	std::uniform_real_distribution<double> dist( -( size * size ), ( size * size ) );
	std::mt19937_64 rng( dev( ) );

	double** rows = new double*[ size ];

	for( auto i = 0; i < size; ++i ){

		rows[ i ] = new double[ size ];

		for( auto j = 0; j < size; ++j ){

			rows[ i ][ j ] = dist( rng );
		}
	}

	return rows;
}

double*& flatten_matrix( double**& matrix, int size ){

	double* flat = new double[ size * size ];

	for( auto i = 0; i < size; ++i ){

		for( auto j = 0; j < size; ++j ){

			flat[ ( i * size ) + j ] = matrix[ i ][ j ];
		}
	}

	return flat;
}

void column_major_layout( double**& matrix, int size ){

	double** copyMatrix = new double*[ size ];

	for( auto i = 0; i < size; ++i ){

		copyMatrix[ i ] = new double[ size ];

		std::memcpy( copyMatrix[ i ], matrix[ i ], size * sizeof( double* ) );
	}

	for( auto i = 0; i < size; ++i ){

		for( auto j = 0; j < size; ++j ){

			matrix[ i ][  j ] = copyMatrix[ j ][ i ];
		}
	}
	
	delete copyMatrix;
}

void print_matrix( double**& matrix, int size ){

	for( auto i = 0; i < size; ++i ){

		for( auto j = 0; j < size; ++j ){

			printf( "%lf ", matrix[ i ][ j ] );
		}

		printf( "\n" );
	}

	printf( "\n" );
}

void print_matrix( double* matrix, int size ){

	for( auto i = 0; i < size; ++i ){

		for( auto j = 0; j < size; ++j ){

			printf( "%lf ", matrix[ ( i * size) + j ] );
		}
		
		printf( "\n" );
	}
}

/// 1. Write a matrix multiplication kernel function that corresponds to the design illustrated in Fig. 6.4.

/// GENERAL IDEIA: the intent is to use Corner Turning. I.e., for two input matrixes A, B, stored in GLOBAL MEMORY, where A is stored in row-major layout, and B in column-major layout, we must
///								 transfer both A and B to the device's SHARED MEMORY in a COALESCED manner. Then, we just proceed with a standard matrix multiplication algorithm, with no need to access
///								 shared memory in coalesced manner.

/// PROBLEM: how the FUCK do I know whether accesses were coalesced or not???????
/// ANSWER: USE THE PROFILER! It tells if accesses were coalesced or not.

/// PROBLEM 2: row-major layout per se is not the problem. The problem arises from the ACCESS PATTERN, i.e., what are we reading from the matrix. If we read a column-major layout ROWISE and perform a
///						  standard multiplication, the dot product for each element of the output matrix is completely wrong, because elements from the same row are actually from the same column. Accessing correct
///						  elements breaks access coalescing. This is the problem!

// ANSWER:

// ASSUMPTIONS: Square matrices only
//__global__ void corner_turning( double*& input_left, double*& input_right, double*& output, int size ) ----> WARNING <---- DO NOT PASS ARGUMENTS BY REFERENCE WITHOUT A VERY GOOD REASON. CAN CAUSE ERRORS!!!!!!!!
__global__ void corner_turning( double* input_left, double* input_right, double* output, int size, int shared_mem_size ){

	/// NOTE: even though ALL THREADS execute the construction of shared memory entities, there is only ONE instance of these entities for ALL threads in a block. I.e., threads do NOT create their own copies of shared entities
	/// NOTE: There are two ways of allocating SHARED memory, STATICALLY and DYNAMICALLY. Static allocation requires a CONSTANT value, either hardcoded or via a constant variable with value known at DEVICE
	///				COMPILE TIME. Dynamical allocation works by declaring a shared entity as extern, i.e., by using the following declaration - extern __shared__ Type var_Name []. The size is implicitly INFERRED from the
	///				LAST parameter passed as an argument to the kernel's execution configuration parameters
	extern __shared__ double left_matrix[ ];
	extern __shared__ double right_matrix[ ];
	
	const int block_IDx{ blockIdx.x }, block_IDy{ blockIdx.y };
	const int thread_X{ threadIdx.x }, thread_Y{ threadIdx.y };

	/// NOTE: I am mistaking the ORIGINAL matrix for its COLUMN-MAJOR version. That is why coalescing is seemingly
	///				"trivial", because converting the original into its column-major version, and then storing the latter in row-
	///				major order, makes coalescing straightforward.
	
	// Setting dimensions of tiles to match block dimensions
	const int tile_width = blockDim.x;

// TILING: represents the indexes that threads in a tile must visit. I.e., tiles are not an entity or a container holding threads. They are just the subsets from input/output data that threads were mapped to. In other words, threads
//				can access only restricted and pre-determined elements from input/ouput matrices.
	for( auto i = 0; i < ( size / tile_width ); ++i ){

		for( auto j = 0; j < ( size / tile_width ); ++j ){

			// EXPLANATION: forget about multidimensional arrays, even when handling one. Keep in mind a flattened array, it makes things easier. Consider an array A with ( size * size ) elements.
			//							 ( i * tile_width * size ) := the correct term here would be tile_height, not width. This expression divides the whole array into ( tile_width * size ) sections, where i is used
			//																		  to identify in which section our threads are. If we had a matrix of tiles, this expression selects the tile ROW.
			//							 ( j * tile_width ) := this expression selects the tile COLUMN, i.e., it subdivides the ( i * tile_width * size ) section into subsections of tile_width size
			//							 ( thread_X * size ) + thread_Y := this expression is used to select an element from within the above subsections. Each thread selects their corresponding elements.
			// int index = ( i * tile_width * size ) + ( j * tile_width ) + ( thread_X * size ) + thread_Y;

			// PROBLEM: somehow, when I update left_matrix, right_matrix gets updated to the SAME value, even though it gets its values from ANOTHER input matrix.  I.E.: address space pollution <<<<---- This is correct!
			// EXPLANATION: even though right_matrix and left_matrix are names bound to different address spaces, both spaces are sequential, i.e., right_matrix ends one address before the adress that holds the first element
			// of left_matrix. But the names themselves, i.e., right_matrix and left_matrix, are	 not dereferenced to their specific spaces, meaning that, using the same index on right_matrix and left_matrix, i.e., doing right_matrix[ i ]
			//							 and left_matrix[ i ], returns us the SAME element, i.e., right_matrix[ i ] = left_matrix[ i ]. Thus, if we want to access the correct elements, we must add a multiple of the offset, corresponding to the size
			// of each array, to each array, in order of declaration. TL;DR: even though we have different names for each shared memory entity, they all refer to the same addresses. Having multiple names only guarantees that the total
			// shared space allocated is a product of the amount of names and the allocation size( passed as an extern parameter )
			
			/// NOTE: we do not know the size of the input matrices. Thus, we also have no way of knowing the size of the output matrices. Then, we must also use tiling on the latter.

			// Here, I am loading the elements in row-major layout, to match the original matrix. This is why I am inverting i,j and thread_X, thread_Y positions with respect to the original index formula. This will make multiplication easier
			// later, since there will be no need to reestructure the matrix.
			right_matrix[ ( j * tile_width * size ) + ( i * tile_width ) + ( thread_Y * size ) + thread_X ] = input_right[ ( i * tile_width * size ) + ( j * tile_width ) + ( thread_X * size ) + thread_Y ]; // Loading to right_matrix in the correct layout
			left_matrix[ ( i * tile_width * size ) + ( j * tile_width ) + ( thread_X * size ) + thread_Y + ( shared_mem_size ) ] = input_left[ ( i * tile_width * size ) + ( j * tile_width ) + ( thread_X * size ) + thread_Y ];

			__syncthreads( );
		}		
	}
		// Let threads in each tile load elements to shared memory

	double dot_product{ 0 };

	int row = ( block_IDy * tile_width ) + thread_Y;
	int col = ( block_IDx * tile_width ) + thread_X;

	for( int i = 0; i < ( size / tile_width ); ++i ){

		for( int j = 0; j < ( size / tile_width ); ++j ){

			for( int k = 0; k < tile_width; ++k ){

				dot_product += left_matrix[ ( i * tile_width * size ) + ( j * tile_width ) + ( thread_X * size ) + k ] * right_matrix[ ( j * tile_width * size ) + ( i * tile_width ) + ( thread_Y * size ) + k ];
			}
			__syncthreads( );

			output[ ( i * tile_width * size ) + ( j * tile_width ) + ( thread_X * size ) + thread_Y ] = dot_product;
		}
	}
}

void kernel_setup( double*& host_left, double*& host_right, double*& host_output, int size ){

	int dimension = ( size * size ) * sizeof( double );

	double* dev_left_in{ nullptr }, *dev_right_in{ nullptr };
	double* dev_out = new double[ dimension ];

	cudaMalloc( ( void** ) &dev_left_in, dimension );
	cudaMalloc( ( void** ) &dev_right_in, dimension );
	cudaMalloc( ( void** ) &dev_out, dimension );

	cudaMemcpy( dev_left_in, host_left, dimension, cudaMemcpyHostToDevice );
	cudaMemcpy( dev_right_in, host_right, dimension, cudaMemcpyHostToDevice );

	print_matrix( host_left, size );

	std::cout << "\n";

	unsigned int num_of_blocks{ 0 };
	unsigned int block_dim{ 0 };

	std::cout << "Enter Grid and Block dimensions: ";
	std::cin >> num_of_blocks >> block_dim;

	dim3 grid_dim( num_of_blocks, 1, 1 );
	dim3 block_size( block_dim, block_dim, 1 );

	int shared_array_size{ size * size };

	host_output = new double[ dimension ];

	corner_turning<<<grid_dim, block_size, shared_array_size * sizeof( double )>>>( dev_left_in, dev_right_in, dev_out, size, shared_array_size );	

	cudaMemcpy( host_output, dev_out, dimension, cudaMemcpyDeviceToHost );

	std::cout << "\n";

	print_matrix( host_output, size );	

	cudaFree( dev_left_in );
	cudaFree( dev_right_in );
}

int main( ){

	int size{ 0 };

	std::cout << "Enter the matrices dimensions: ";
	std::cin >> size;

	std::cout << "\n";

	double** left_matrix = generate_matrix( size );
	double** right_matrix = generate_matrix( size );

	print_matrix( right_matrix, size );

	//std::cout << "\n";

	column_major_layout( right_matrix, 4 );

	double* flat_left = flatten_matrix( left_matrix, size );
	double* flat_right = flatten_matrix( right_matrix, size );

	double* output{ nullptr };

	//print_matrix( flat_right, 4 );

	//std::cout << "\n";

	kernel_setup( flat_left, flat_right, output, size );
}