#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <random>
#include <bitset>

unsigned int*& generate_array( int size ) {

	unsigned int* array = new unsigned int[ size ];

	std::random_device dev;
	std::uniform_int_distribution<int> dist( -( size * size ), ( size * size ) );
	std::mt19937_64 rng( dev( ) );

	for( auto i = 0; i < size; ++i ) {

		array[ i ] = dist( rng );
	}

	return array;
}

void print( unsigned int*& array, int size ) {

	for( auto i = 0; i < size; ++i ) {

		std::cout << std::bitset<32>( array[ i ] ) << " ";
	}

	printf( "\n" );
}

/// PROBLEM: 1. Extend the kernel in Fig. 13.4 by using shared memory to improve memory coalescing.

/// ANSWER:

// NOTE: I am  using the Brent-Kung Scan Algorithm here, but as an Exclusive Scan
__global__ void exclusiveScan( unsigned int* bits, unsigned int* output, unsigned int N, unsigned int Section_Size, int* flags, int* scan_value, int* blockCounter ) {

	__shared__ unsigned int bid_s;

	if( threadIdx.x == 0 ) {

		bid_s = atomicAdd( blockCounter, 1 );
	}

	__syncthreads( );

	unsigned int bid = bid_s;

	__shared__ extern unsigned int Shared_Input[ ];

	unsigned int i = ( 2 * blockIdx.x * blockDim.x ) + threadIdx.x;

	if( ( i < N ) && ( threadIdx.x != 0 ) ) {

		Shared_Input[ threadIdx.x ] = bits[ i - 1 ];
	}

	if( ( i + blockDim.x ) < N ) {

		Shared_Input[ threadIdx.x + blockDim.x ] = bits[ i + blockDim.x ];
	}

	for( unsigned int stride = 1; stride <= blockDim.x; stride *= 2 ) {

		__syncthreads( );

		unsigned int index = ( ( threadIdx.x + 1 ) * 2 * stride ) - 1;

		if( index < Section_Size ) {

			Shared_Input[ index ] += Shared_Input[ index - stride ];
		}
	}

	int correct_rounding{ 0 };

	// EXPLANATION: stride must be in the form of Powers of 2, but the code on the book allows for values that are not powers of 2. Thus, we pick the smallest power of 2 that is
	//				greater than the quotient Section_Size / 4.
	correct_rounding = static_cast< int >( pow( 2.0, ceil( log2( static_cast< double >( Section_Size ) / 4.0 ) ) ) );

	for( int stride = correct_rounding; stride > 0; stride /= 2 ) {

		__syncthreads( );

		unsigned int index = ( ( threadIdx.x + 1 ) * stride * 2 ) - 1;

		if( ( index + stride ) < Section_Size ) {

			Shared_Input[ index + stride ] += Shared_Input[ index ];
		}
	}

	__syncthreads( );

	if( i < N ) {

		unsigned int holder = Shared_Input[ threadIdx.x ];

		output[ i ] = holder;
	}

	if( ( i + blockDim.x ) < N ) {

		unsigned int holder = Shared_Input[ threadIdx.x + blockDim.x ];

		output[ i + blockDim.x ] = holder;
	}

	// Block Synchronization

	__shared__ unsigned int previous_sum;

	if( threadIdx.x == 0 ) {

		// bid = blockId
		while( atomicAdd( &flags[ bid ], 0 ) == 0 ) { }

		previous_sum = scan_value[ bid ];

		// TODO: what the fuck is local_sum
		scan_value[ bid + 1 ] = previous_sum + output[ i + blockDim.x ];

		__threadfence( );

		atomicAdd( &flags[ bid + 1 ], 1 );
	}

	__syncthreads( );
}

__global__ void radix_sort_iter( unsigned int* input, unsigned int* output, unsigned int* bits, unsigned int N, unsigned int iter, int* flags, int* scan_value, int* blockCounter ) {

	__shared__ extern unsigned int shared_bits[ ];

	unsigned int i = ( blockIdx.x * blockDim.x ) + threadIdx.x;

	unsigned int key{ 0 }, bit{ 0 };

	if( i < N ) {

		key = input[ i ];
		bit = ( key >> iter ) & 1;
		shared_bits[ i ] = bit;
	}

	printf( "test" );

	__syncthreads( );

	// Count the amount of 1's before i
	exclusiveScan<<<gridDim.x, blockDim.x, blockDim.x * sizeof( unsigned int ) >> > ( shared_bits, output, N, ( N / gridDim.x ), flags, scan_value, blockCounter );

	if( i < N ) {

		unsigned int OnesBefore = shared_bits[ i ];
		unsigned int OnesTotal = shared_bits[ N ]; // This means that shared memory MUST have N elements
		unsigned int dst = ( bit == 0 ) ? ( i - OnesBefore ) : ( N - OnesTotal - OnesBefore );

		output[ dst ] = key;
	}
}

void kernel_setup( unsigned int* host_input, unsigned int* host_output, unsigned int* host_bits, unsigned int host_N, unsigned host_iter, int size ) {

	unsigned int* dev_input{ nullptr }, * dev_output{ nullptr }, * dev_bits{ nullptr };
	unsigned int dev_N{ 0 }, dev_iter{ 0 };
	int* scan_value{ 0 }, * block_counter{ 0 };
	int* flags{ false }, * dev_flags;

	unsigned int array_size = size * sizeof( unsigned int );
	unsigned int N_size = host_N * sizeof( unsigned int );
	unsigned int iter_size = host_iter * sizeof( unsigned int );

	cudaMalloc( ( void** ) &dev_input, array_size );
	cudaMalloc( ( void** ) &dev_output, array_size );
	cudaMalloc( ( void** ) &dev_bits, array_size ); // ATTENTION: potentially incorrect size
	cudaMalloc( ( void** ) &dev_N, N_size );
	cudaMalloc( ( void** ) &dev_iter, iter_size );

	cudaMemcpy( dev_input, host_input, array_size, cudaMemcpyHostToDevice );
	cudaMemcpy( dev_bits, host_bits, array_size, cudaMemcpyHostToDevice );
	cudaMemcpy( &dev_N, &host_N, N_size, cudaMemcpyHostToDevice );
	cudaMemcpy( &dev_iter, &host_iter, iter_size, cudaMemcpyHostToDevice );

	unsigned int num_blocks{ 0 };
	unsigned int num_threads{ 0 };

	std::cout << "\nEnter the number of blocks: ";
	std::cin >> num_blocks;

	std::cout << "\nEnter the number of threads: ";
	std::cin >> num_threads;

	cudaMalloc( ( void** ) &scan_value, num_blocks * sizeof( unsigned int ) );
	cudaMalloc( ( void** ) &dev_flags, num_blocks * sizeof( int ) );
	cudaMalloc( ( void** ) &block_counter, sizeof( int ) );

	flags = new int[ num_blocks ];

	flags[ 0 ] = 1;

	for( auto i = 1; i < num_blocks; ++i ) {

		flags[ i ] = 0;
	}

	cudaMemcpy( dev_flags, flags, num_blocks * sizeof( int ), cudaMemcpyHostToDevice );

	dim3 blocks{ num_blocks };
	dim3 threads{ num_threads };

	unsigned int shared_memsize{ 0 };

	std::cout << "\nEnter the size of shared memory: ";
	std::cin >> shared_memsize;

	shared_memsize *= sizeof( unsigned int );

	radix_sort_iter<<<blocks, threads, shared_memsize>>>( dev_input, dev_output, dev_bits, dev_N, dev_iter, dev_flags, scan_value, block_counter );

	cudaFree( dev_input );
	cudaFree( dev_output );
	cudaFree( dev_bits );
	cudaFree( &dev_N );
	cudaFree( &dev_iter );
	cudaFree( scan_value );
	cudaFree( block_counter );
	cudaFree( dev_flags );
}

int main( ) {

	int size{ 0 };

	std::cout << "Enter the size of the array: ";
	std::cin >> size;

	unsigned int* array = generate_array( size );
	unsigned int* output = new unsigned int[ size ];
	unsigned int* bits = new unsigned int[ size ];

	print( array, size );

	kernel_setup( array, output, bits, size, size, size );

	print( output, size );

	delete[ ] array;
}