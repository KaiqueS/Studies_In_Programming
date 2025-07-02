#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <random>

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

void print( int*& array, int size ){

	for( auto i = 0; i < size; ++i ){

		printf( "%d ", array[ i ] );
	}

	printf( "\n" );
}

/// PROBLEM: 1. Extend the kernel in Fig. 13.4 by using shared memory to improve memory coalescing.

/// ANSWER:

__global__ void exclusiveScan( unsigned int* bits, unsigned int N ){

}

__global__ void radix_sort_iter( unsigned int* input, unsigned int* output, unsigned int* bits, unsigned int N, unsigned int iter ){

	__shared__ extern int shared[ ];

	unsigned int i = ( blockIdx.x * blockDim.x ) + threadIdx.x;

	unsigned int key{ 0 }, bit{ 0 };

	if( i < N ){

		key = input[ i ];
		bit = ( key >> iter ) & 1;
		bits[ i ] = bit;
	}

	exclusiveScan<<<gridDim.x, blockDim.x>>>( bits, N );

	if( i < N ){

		unsigned int OnesBefore = bits[ i ];
		unsigned int OnesTotal = bits[ N ];
		unsigned int dst = ( bit == 0 ) ? ( i - OnesBefore ) : ( N - OnesTotal - OnesBefore );

		output[ dst ] = key;
	}
}

void kernel_setup( unsigned int* host_input, unsigned int* host_output, unsigned int* host_bits, unsigned int host_N, unsigned host_iter, int size ){

	unsigned int* dev_input{ nullptr }, * dev_output{ nullptr }, * dev_bits{ nullptr };
	unsigned int dev_N{ 0 }, dev_iter{ 0 };

	unsigned int array_size = size * sizeof( int );
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

	dim3 blocks{ num_blocks };
	dim3 threads{ num_threads };

	unsigned int shared_memsize{ 0 };

	std::cout << "\nEnter the size of shared memory: ";
	std::cin >> shared_memsize;

	shared_memsize *= sizeof( unsigned int );

	radix_sort_iter<<<blocks, threads, shared_memsize>>>( dev_input, dev_output, dev_bits, dev_N, dev_iter );

	cudaFree( dev_input );
	cudaFree( dev_output );
	cudaFree( dev_bits );
	cudaFree( &dev_N );
	cudaFree( &dev_iter );
}

int main( ){

	int size{ 0 };

	std::cout << "Enter the size of the array: ";
	std::cin >> size;

	int* array = generate_array( size );

	print( array, size );

	delete[ ] array;
}