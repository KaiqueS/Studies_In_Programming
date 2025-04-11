#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <random>

double*& generate_array( int size ){

	std::random_device dev;
	std::uniform_real_distribution<double> dist( -( size * size ), ( size * size ) );
	std::mt19937_64 rng( dev( ) );

	double* array = new double[ size ];

	for( auto i = 0; i < size; ++i ){

		array[ i ] = dist( rng );
	}

	return array;
}

void print( double*& array, int size ){

	for( auto i = 0; i < size; ++i ){

		std::cout << array[ i ] << ' ';
	}

	std::cout << "\n";
}

/// PROBLEM: 3. Modify the kernel in Fig. 10.9 to use the access pattern illustrated below.

/// ANSWER: 

__global__ void ConvergentSumReductionKernel( double* input, double* output ){

	unsigned int i = threadIdx.x;

	for( unsigned int stride = 0; stride < blockDim.x; stride /= 2 ){

		if( threadIdx.x >= stride ){

			input[ i ] += input[ i + stride ];

			//printf( "%d, %d ", stride, threadIdx.x );
		}

		__syncthreads( );
	}

	if( threadIdx.x == ( blockDim.x - 1 ) ){

		*output = input[ blockDim.x - 1 ];
	}
}

void kernel_setup( double* host_input, double* host_output, int input_size ){

	double* dev_input{ nullptr }, *dev_output{ nullptr };

	int dev_size = input_size * sizeof( double );

	cudaMalloc( ( void** ) &dev_input, dev_size );
	cudaMalloc( ( void** ) &dev_output, ( 1 * sizeof( double ) ) );

	cudaMemcpy( dev_input, host_input, dev_size, cudaMemcpyHostToDevice );

	unsigned int thread_number = std::floor( static_cast<double>( input_size ) / 2.0 ); // ATTENTION: rounding errors

	dim3 block( 1 );
	dim3 threads{ thread_number };

	ConvergentSumReductionKernel<<<block, threads>>>( dev_input, dev_output );

	cudaMemcpy( host_output, dev_output, ( 1 * sizeof( double ) ), cudaMemcpyDeviceToHost );

	cudaFree( dev_input );
	cudaFree( dev_output );
}

int main( ){

	int size{ 0 };

	std::cout << "Enter the size of the array: ";
	std::cin >> size;

	double* array = generate_array( size );
	double output{ 0.0 };

	std::cout << "\n";

	print( array, size );

	kernel_setup( array, &output, size );

	std::cout << "\n" << output << "\n";

	delete[ ] array;
}