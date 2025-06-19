#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <random>
#include <math.h>
#include <algorithm>

struct Pair{

	int first{ 0 };
	int second{ 0 };
};

int*& generate_array( int size ){

	int* array = new int[ size ];

	std::random_device dev;
	std::uniform_int_distribution<int> dist( -( size * size ), ( size * size ) );
	std::mt19937_64 rng( dev( ) );

	for( auto i = 0; i < size; ++i ){

		array[ i ] = dist( rng );
	}

	return array;
}

Pair split_array( int* input, int*& left, int*& right, int size ){

	Pair values{ };

	int split = ( size % 2 == 0 ) ? ( size / 2 ) : static_cast<int>( std::ceil( static_cast<double>( size ) / 2.0 ) );

	int remainder = size - split;

	values.first = split;
	values.second = remainder;

	left = new int[ split ];
	right = new int[ remainder ];

	for( auto i = 0; i < split; ++i ){

		left[ i ] = input[ i ];

		if( i < remainder ){

			right[ i ] = input[ split + i ];
		}

		else{

			continue;
		}
	}

	return values;
}

void print_array( int*& array, int size ){

	for( auto i = 0; i < size; ++i ){

		printf( "%d ", array[ i ] );
	}

	printf( "\n" );
}

// FOR TESTING ONLY. THIS IS INEFFICIENT
void linear_sort( int*& array, int size ){

	int holder{ 0 };

	for( auto i = 0; i < size; ++i ){

		for( auto j = i + 1; j < size; ++j ){

			if( array[ i ] > array[ j ] ){

				holder = array[ i ];

				array[ i ] = array[ j ];
				array[ j ] = holder;
			}

			else{

				continue;
			}
		}
	}
}

// NOTE: A and B must already be sorted AT FUNCTION CALL!
__global__ void merge_sequential( int* A, int m, int* B, int n, int* C ){

	int i{ 0 }, j{ 0 }, k{ 0 };

	while( ( i < m ) && ( j < n ) ){

		C[ k++ ] = ( A[ i ] <= B[ j ] ) ? A[ i++ ] : B[ j++ ];
	}

	if( i == m ){

		while( j < n ){

			C[ k++ ] = B[ j++ ];
		}
	}

	else{

		while( i < m ){

			C[ k++ ] = A[ i++ ];
		}
	}
}

// NOTE: I tried letting co_rank be a __global__, instead of a __device__, function. What I learned from it?
//		 1- __global__ does not allow values to be returned from the kernel
//		 2- __global__ REQUIRES execution configuration parameters <<<>>>>
//		 3- __device__ REQUIRES returning values, but does NOT need execution configuration parameters
__device__ int co_rank( int k, int* A, int m, int* B, int n ){

	int i = ( k < m ) ? k : m;
	int j = k - i;
	int i_low = 0 > ( k - n ) ? 0 : ( k - n );
	int j_low = 0 > ( k - m ) ? 0 : ( k - m );
	int delta{ 0 };

	bool active = true;

	while( active ){

		if( ( i > 0 ) && ( j < n ) && ( A[ i - 1 ] > B[ j ] ) ){

			delta = ( ( i - i_low + 1 ) >> 1 );
			j_low = j;
			j += delta;
			i -= delta;
		}

		else if( ( j > 0 ) && ( i < m ) && ( B[ j - i ] >= A[ i ] ) ){

			delta = ( ( j - j_low + 1 ) >> 1 );
			i_low = i;
			i += delta;
			j -= delta;
		}

		else{

			active = false;
		}
	}

	return i;
}

__global__ void merge_basic_kernel( int* A, int m, int* B, int n, int* C ){

	int tid = ( blockIdx.x * blockDim.x ) + threadIdx.x;
	int elementsPerThread = static_cast<int>( ceil( static_cast<double>( m + n ) / static_cast<double>( blockDim.x * gridDim.x ) ) );
	
	int k_curr = ( tid * elementsPerThread );
	int k_next = static_cast<int>( min( ( ( tid + 1 ) * elementsPerThread ), ( m + n ) ) );

	int i_curr = co_rank( k_curr, A, m, B, n );
	int i_next = co_rank( k_next, A, m, B, n );
	int j_curr = k_curr - i_curr;
	int j_next = k_next - i_next;

	// NOTE: to be able to call a __global__ kernel from within another kernel, some changes must be made to the project properties.
	//		 Follow: https://stackoverflow.com/a/64531164/10286737 to correctly set up things.
	merge_sequential<<<gridDim.x, blockDim.x>>>( &A[ i_curr ], ( i_next - i_curr ), &B[ j_curr ], ( j_next - j_curr ), &C[ k_curr ] );
}

__global__ void merge_tiled_kernel( int* A, int m, int* B, int n, int* C, int tile_size ){


}

__global__ void merge_ciruclar_buffer_kernel( int* A, int m, int* B, int n, int* C ){


}


void kernel_setup( int* host_A, int host_m, int* host_B, int host_n, int* host_C ){

	int* dev_A{ nullptr }, *dev_B{ nullptr }, *dev_C{ nullptr };

	int A_size = host_m * sizeof( int );
	int B_size = host_n * sizeof( int );
	int C_size = ( host_m + host_n ) * sizeof( int );

	cudaMalloc( ( void** ) &dev_A, A_size );
	cudaMalloc( ( void** ) &dev_B, B_size );
	cudaMalloc( ( void** ) &dev_C, C_size );

	cudaMemcpy( dev_A, host_A, A_size, cudaMemcpyHostToDevice );
	cudaMemcpy( dev_B, host_B, B_size, cudaMemcpyHostToDevice );

	unsigned int num_blocks{ 0 };
	unsigned int num_threads{ 0 };

	std::cout << "Enter the number of blocks: ";
	std::cin >> num_blocks;

	std::cout << "\nEnter the number of threads: ";
	std::cin >> num_threads;

	dim3 blocks{ num_blocks, 1, 1 };
	dim3 threads{ num_threads };

	merge_basic_kernel<<<blocks, threads>>>( dev_A, host_m, dev_B, host_n, dev_C );

	cudaMemcpy( host_C, dev_C, C_size, cudaMemcpyDeviceToHost );

	cudaFree( dev_A );
	cudaFree( dev_B );
	cudaFree( dev_C );
}

int main( ){

	int size{ 0 };

	std::cout << "Enter the size of the array: ";
	std::cin >> size;

	int* array = generate_array( size );
	int* left{ nullptr }, *right{ nullptr };
	int* output = new int[ size ];

	Pair values = split_array( array, left, right, size );

	print_array( array, size );

	linear_sort( left, values.first );
	linear_sort( right, values.second );

	std::cout << "\n";

	kernel_setup( left, values.first, right, values.second, output );

	print_array( output, size );

	delete[ ] array, left, right, output;
}