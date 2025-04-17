#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <math.h>
#include <random>

double*&& generate_array( int size ){

	std::random_device dev;
	std::uniform_real_distribution<double> dist( -( size * size ), ( size * size ) );
	std::mt19937_64 rng( dev( ) );

	double* array = new double[ size ];

	for( auto i = 0; i < size; ++i ){

		array[ i ] = dist( rng );
	}

	return std::move( array );
}

void print( double*& array, int size ){

	for( auto i = 0; i < size; ++i ){

		std::cout << array[ i ] << " ";
	}

	std::cout << "\n";
}

/// PROBLEM - 5. Modify the kernel in Fig. 10.15 to work for an arbitrary length input that is not necessarily a multiple of COARSE_FACTOR * 2 * blockDim.x. Add an extra
///				 parameter N to the kernel that represents the length of the input.

/// ANSWER: Instead of adding a parameter to the Kernel, I modified the amount of threads so as to ALWAYS launch the minimum amount of threads necessary to process an input

__global__ void CoarsenedSumReductionKernel( double* input, double* output, unsigned int Coarse_Factor ){

	__shared__ extern double shared_input[ ]; // Shared-Memory size := block dimension BY CONSTRUCTION at kernel call through execution configuration parameters

	unsigned int segment = Coarse_Factor * 2 * blockDim.x * blockIdx.x; // NOTE: multiply by 2 because we are performing a binary operation, i.e., requires 2 elements
	unsigned int i = segment + threadIdx.x;
	unsigned int t = threadIdx.x;

	double sum = input[ i ];

	for( unsigned int tile = 1; tile < ( Coarse_Factor * 2 ); ++tile ){

		sum += input[ ( blockDim.x * tile ) + i ];
	}

	shared_input[ t ] = sum;
	
	for( unsigned int stride = ( blockDim.x / 2 ); stride >= 1; stride /= 2 ){

		__syncthreads( );

		if( t < stride ){

			shared_input[ t ] += shared_input[ t + stride ];
		}
	}

	if( t == 0 ){

		atomicAdd( output, shared_input[ 0 ] );
	}
}

void kernel_setup( double*& host_array, double& output, int size ){

	double* dev_array{ nullptr }, *dev_out{ nullptr };

	int array_size = size * sizeof( double );

	cudaMalloc( ( void** ) &dev_array, array_size );
	cudaMalloc( ( void** ) &dev_out, 1 * sizeof( double ) );

	cudaMemcpy( dev_array, host_array, array_size, cudaMemcpyHostToDevice );
	cudaMemcpy( dev_out, &output, 1 * sizeof( double ), cudaMemcpyHostToDevice );

	dim3 blocks{ 1, 1, 1 };

	unsigned int coarse_factor{ 0 };

	std::cout << "\nEnter the Coarse Factor: ";
	std::cin >> coarse_factor;

	unsigned int num_threads{ static_cast<unsigned int>( std::ceil( static_cast<double>( size ) / static_cast<double>( 2 * coarse_factor ) ) ) }; // Remember that atomicAdd is a binary operation, i.e., involves 2 elements. Thus, 2 * coarse_factor 

	dim3 threads{ num_threads, 1, 1 };

	int shared_memsize = num_threads * sizeof( double );

	CoarsenedSumReductionKernel<<<blocks, num_threads, shared_memsize>>>( dev_array, dev_out, coarse_factor );

	cudaMemcpy( &output, dev_out, 1 * sizeof( double ), cudaMemcpyDeviceToHost );

	cudaFree( dev_array );
	cudaFree( dev_out );
}

int main( ){

	int size{ 0 };

	std::cout << "Enter the size of the array: ";
	std::cin >> size;

	std::cout << "\n";

	double* array = generate_array( size );
	double output{ 0 };

	print( array, size );

	kernel_setup( array, output, size );

	std::cout << "\n" << output << "\n";

	delete[ ] array;
}