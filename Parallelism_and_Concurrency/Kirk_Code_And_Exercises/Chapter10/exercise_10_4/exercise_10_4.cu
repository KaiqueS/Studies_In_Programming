#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <math.h>
#include <random>

int*&& generate_array( int size ){

	std::random_device dev;
	std::uniform_int_distribution<int> dist( -( size * size ), ( size * size ) );
	std::mt19937_64 rng( dev( ) );

	int* array = new int[ size ];

	for( auto i = 0; i < size; ++i ){

		array[ i ] = dist( rng );
	}

	return std::move( array );
}

void print( int*& array, int size ){

	for( auto i = 0; i < size; ++i ){

		std::cout << array[ i ] << " ";
	}

	std::cout << "\n";
}

/// PROBLEM: 4. Modify the kernel in Fig. 10.15 to perform a max reduction instead of a sum reduction.

/// ANSWER: 

__global__ void CoarsenedSumReductionKernel( int* input, int* output, unsigned int Coarse_Factor ){

	__shared__ extern int shared_input[ ]; // Shared-Memory size := block dimension BY CONSTRUCTION at kernel call through execution configuration parameters

	unsigned int segment = Coarse_Factor * 2 * blockDim.x * blockIdx.x; // NOTE: multiply by 2 because we are performing a binary operation, i.e., requires 2 elements
	unsigned int i = segment + threadIdx.x;
	unsigned int t = threadIdx.x;

	int greatest = input[ i ];

	for( unsigned int tile = 1; tile < ( Coarse_Factor * 2 ); ++tile ){

		greatest = ( greatest > input[ ( tile * blockDim.x ) + i ] ) ? greatest : input[ ( tile * blockDim.x ) + i ];
	}

	shared_input[ t ] = greatest;
	
	for( unsigned int stride = ( blockDim.x / 2 ); stride >= 1; stride /= 2 ){

		__syncthreads( );

		if( t < stride ){

			shared_input[ t ] = ( shared_input[ t ] > shared_input[ t + stride ] ) ? shared_input[ t ] : shared_input[ t + stride ];
		}
	}

	if( t == 0 ){

		atomicMax( output, shared_input[ 0 ] );
	}
}

void kernel_setup( int*& host_array, int& output, int size ){

	int* dev_array{ nullptr }, *dev_out{ nullptr };

	int array_size = size * sizeof( int );

	cudaMalloc( ( void** ) &dev_array, array_size );
	cudaMalloc( ( void** ) &dev_out, 1 * sizeof( int ) );

	cudaMemcpy( dev_array, host_array, array_size, cudaMemcpyHostToDevice );
	cudaMemcpy( dev_out, &output, 1 * sizeof( int ), cudaMemcpyHostToDevice );

	dim3 blocks{ 1, 1, 1 };

	unsigned int coarse_factor{ 0 };

	std::cout << "\nEnter the Coarse Factor: ";
	std::cin >> coarse_factor;

	unsigned int num_threads{ static_cast<unsigned int>( size ) / ( 2 * coarse_factor ) }; // Remember that atomicMax is a binary operation, i.e., involves 2 elements. Thus, 

	dim3 threads{ num_threads, 1, 1 };

	int shared_memsize = num_threads * sizeof( int );

	CoarsenedSumReductionKernel<<<blocks, num_threads, shared_memsize>>>( dev_array, dev_out, coarse_factor );

	cudaMemcpy( &output, dev_out, sizeof( int ), cudaMemcpyDeviceToHost );

	cudaFree( dev_array );
	cudaFree( dev_out );
}

int main( ){

	int size{ 0 };

	std::cout << "Enter the size of the array: ";
	std::cin >> size;

	std::cout << "\n";

	int* array = generate_array( size );
	int output{ 0 };

	print( array, size );

	kernel_setup( array, output, size );

	std::cout << "\n" << output << "\n";

	delete[ ] array;
}