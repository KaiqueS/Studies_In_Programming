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

		//std::cout << std::bitset<32>( array[ i ] ) << " ";

		printf( "%d ", array[ i ] );
	}

	printf( "\n" );
}

/// PROBLEM: Extend the kernel in Fig. 13.4 by using shared memory to improve memory coalescing.

/// ANSWER:

/// TODO: USE DEBUGGER!!!!!!!!!!!!!!!!!!!!!!!!!

// NOTE/PROBLEM: shared memory, even in nested kernel calls, does not persist through calls. Thus, passing a shared memory array as argument to the
//				 unsigned int* bits parameters results in bits, within exclusiveScan, being different from the shared memory array that initializes
//				 it. I.e., bits, within exclusiveScan, != its argument.
__device__ void exclusiveScan( unsigned int* bits, unsigned int* output, unsigned int N, int* flags, int* scan_value, int blockCounter ) {

	// NOTE: potential cause of the problem
	__shared__ unsigned int bid_s;

	if( threadIdx.x == 0 ) {

		bid_s = atomicAdd( &blockCounter, 1 );
	}

	__syncthreads( );

	unsigned int bid = bid_s;

	// NOTE: potential cause of the problem
	// NOTE: SharedMem has size equal to blockDim.x, i.e., it only stores the partial sum of each block
	__shared__ extern unsigned int SharedMem[ ];

	unsigned int i = ( blockDim.x * blockIdx.x ) + threadIdx.x;

	// NOTE: since we are adding up bits, we default to 0 if the following condition is not met, because
	//		 0 is the addition identity
	// PROBLEM: let bits = [ 0, 0, 1, 1 ]. SharedMem should be SharedMem = [ 0, 1, 1, 0 ], but it is
	//			SharedMem = [ 0, 0, 0, 1 ] instead. This is WRONG
	SharedMem[ threadIdx.x ] = ( ( i < N ) && ( threadIdx.x != 0 ) ) ? bits[ i - 1 ] : 0;

	__syncthreads( );

	//printf( "%d ", SharedMem[ threadIdx.x ] );

	// NOTE: what the hell is this adding up? We should not add up bits, we should count them.
	//		 No, we are actually adding them.
	for( unsigned int stride = 1; stride < blockDim.x; stride *= 2 ){

		__syncthreads( );

		unsigned int temp{ };

		if( threadIdx.x >= stride ){

			// NOTE: using threadIdx.x as an index might be WRONG
			//		 hmmmmm, I do not think so. SharedMem is block-local, and its size equals the size of the block,
			//		 thus, threadIdx.x is always within range
			temp = SharedMem[ threadIdx.x ] + SharedMem[ threadIdx.x - stride ];
		}

		__syncthreads( );

		if( threadIdx.x >= stride ){

			// NOTE: using threadIdx.x as an index might be WRONG
			SharedMem[ threadIdx.x ] = temp;
		}
	}

	__syncthreads( );

	if( threadIdx.x == 0 ){
		
		// NOTE: scan_value has gridDim.x elements, thus bid is always in the correct range
		scan_value[ bid ] = SharedMem[ blockDim.x - 1 ];
	}

	// Block Synchronization

	// NOTE: potential cause of the problem
	__shared__ unsigned int previous_sum;

	/*if( i == 0 && threadIdx.x == 0 ){

		printf( "%#010x, %#010x, %#010x\n", &bid_s, &SharedMem, &previous_sum );
	}*/

	if( threadIdx.x == 0 ) {

		// bid = blockId
		while( atomicAdd( &flags[ bid ], 0 ) == 0 ) { }

		// PROBLEM: where the FUCK was scan_value previously initialized for this to even make sense?
		previous_sum = scan_value[ bid ];

		// NOTE: substituted output[] for bits[]
		scan_value[ bid + 1 ] = previous_sum + bits[ i + blockDim.x ];

		__threadfence( );

		atomicAdd( &flags[ bid + 1 ], 1 );
	}

	__syncthreads( );
}

// SOLUTION DESCRIPTION:
//							for an input of size N, assign ( N / BLOCKS ) elements from the input to each block
//							assign each element from the block-subset to a thread in the block
//							each thread must extract the LSB from their corresponding element
//								each thread must store their bit into the bits array in the global memory
//							perform an exclusive scan over the bits array, count the amount of 1's in the array
__global__ void radix_sort_iter( unsigned int* input, unsigned int* output, unsigned int* bits, unsigned int N, unsigned int iter, int* flags, int* scan_value, int blockCounter ) {

	unsigned int i = ( blockIdx.x * blockDim.x ) + threadIdx.x;

	unsigned int key{ 0 }, bit{ 0 };

	if( i < N ) {

		key = input[ i ];
		bit = ( key >> iter ) & 1;
		bits[ i ] = bit;

		//printf( "%d ", bits[ i ] );
	}

	__syncthreads( );

	// Counts the amount of 1's before i
	// PROBLEM: exclusiveScan is NOT counting the amount of 0/1 before the ith position
	// PROBLEM: I think that grid-synchronization is messing up with i = ( blockDim.x * blockId.x ) + threadIdx.x!!! -> WRONG. Following comment is right!
	//			Or... each thread calls exclusiveScan. EACH call launches a grid with gridDim blocks, each with blockDim threads.
	//			This dynamic parallelism is fucking things up. <- This is right!
	// NOTE: even though __device__ takes no execution configuration parameters, the argument passed to the call to radix_sort_iter
	//		 is also passed to whatever calls extern within exclusiveScan
	//		 Or not... I removed it and the code still worked. Weird. Try to figure out why.
	exclusiveScan( bits, output, N, flags, scan_value, blockCounter );

	if( i < N ){

		unsigned int OnesBefore = bits[ i ];
		unsigned int OnesTotal = bits[ N - 1 ]; // This means that bits MUST have N elements
		unsigned int dst = ( bit == 0 ) ? ( i - OnesBefore ) : ( N - OnesTotal - OnesBefore );

		//printf( "%d %d ", OnesBefore, OnesTotal );

		output[ dst ] = key;
	}
}

void kernel_setup( unsigned int* host_input, unsigned int* host_output, unsigned int* host_bits, unsigned int host_N ) {

	unsigned int* dev_input{ nullptr }, *dev_output{ nullptr }, *dev_bits{ nullptr };
	
	int* flags{ nullptr }, *dev_flags{ nullptr };
	int* scan_value{ nullptr };

	// NOTE: block_counter is NOT AN ARRAY
	int block_counter{ 0 };
	
	unsigned int array_size = host_N * sizeof( unsigned int );

	cudaMalloc( ( void** ) &dev_input, array_size );
	cudaMalloc( ( void** ) &dev_output, array_size );
	cudaMalloc( ( void** ) &dev_bits, array_size ); // ATTENTION: potentially incorrect size

	cudaMemcpy( dev_input, host_input, array_size, cudaMemcpyHostToDevice );

	unsigned int num_blocks{ 0 };
	unsigned int num_threads{ 0 };

	std::cout << "\nEnter the number of blocks: ";
	std::cin >> num_blocks;

	cudaMalloc( ( void** ) &scan_value, num_blocks * sizeof( int ) );

	std::cout << "\nEnter the number of threads: ";
	std::cin >> num_threads;

	cudaMalloc( ( void** ) &dev_flags, num_blocks * sizeof( int ) );

	// NOTE: flags used to indicate which block to run next. If flags[ i ] != 0, for any i, it is i's turn to run.
	//		 After i's turn, the block i sets flags[ i + 1 ] to 1, so block i + 1 can run.
	flags = new int[ num_blocks ];

	flags[ 0 ] = 1;

	for( auto i = 1; i < num_blocks; ++i ) {

		flags[ i ] = 0;
	}

	cudaMemcpy( dev_flags, flags, num_blocks * sizeof( int ), cudaMemcpyHostToDevice );

	dim3 blocks{ num_blocks };
	dim3 threads{ num_threads };

	unsigned int shared_memsize{ 0 };

	//std::cout << "\nEnter the size of shared memory: ";
	//std::cin >> shared_memsize;

	shared_memsize = num_threads * sizeof( unsigned int );

	for( auto iter = 0; iter < ( 8 * sizeof( unsigned int ) ); ++iter ){

		// NOTE: potential cause of the problem shared_memsize might be incorrect
		radix_sort_iter<<<blocks, threads, shared_memsize>>>( dev_input, dev_output, dev_bits, host_N, iter, dev_flags, scan_value, block_counter );
	}

	cudaMemcpy( host_output, dev_output, array_size, cudaMemcpyDeviceToHost );

	cudaFree( dev_input );
	cudaFree( dev_output );
	cudaFree( dev_bits );
	cudaFree( dev_flags );
	cudaFree( scan_value );

	delete[ ] flags;
}

// NOTE: Parallel Radix Sort requires calling the sorting kernel ITERATEDLY, where the amount of iterations ranges from 0 to the size in bit of each input
int main( ) {

	int size{ 0 };

	std::cout << "Enter the size of the array: ";
	std::cin >> size;

	unsigned int* array = generate_array( size );
	unsigned int* output = new unsigned int[ size ];
	unsigned int* bits = new unsigned int[ size ];

	print( array, size );

	kernel_setup( array, output, bits, size );

	std::cout << "\n";

	print( output, size );

	delete[ ] array, output, bits;
}