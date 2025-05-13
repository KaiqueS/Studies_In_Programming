#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <random>

double*& generate_array( int size ){

	double* array = new double[ size ];

	std::random_device dev;
	std::uniform_real_distribution<double> dist( -( size * size ), ( size * size ) );
	std::mt19937_64 rng( dev( ) );

	for( auto i = 0; i < size; ++i ){

		array[ i ] = dist( rng );
	}

	return array;
}

void print_array( double*& array, int size ){

	for( auto i = 0; i < size; ++i ){

		std::cout << array[ i ] << " ";
	}

	std::cout << "\n";
}

/// PROBLEM: 7. Use the algorithm in Fig. 11.4 to complete an exclusive scan kernel.

/// ANSWER: 

__global__ void Kogge_Stone_Exclusive_Kernel( double* input, double* output, unsigned int input_size ){

	__shared__ extern double SharedMem[ ];

	unsigned int i = ( blockDim.x * blockIdx.x ) + threadIdx.x;

	// NOTE: this is the answer -> Add the identity 0 to the start of the array, and move all other elements one position to the right.
	SharedMem[ threadIdx.x ] = ( ( i < input_size ) && ( threadIdx.x != 0 ) ) ? input[ i - 1 ] : 0.0;

	for( unsigned int stride = 1; stride < blockDim.x; stride *= 2 ){

		__syncthreads( );

		double temp{ };

		if( threadIdx.x >= stride ){

			temp = SharedMem[ threadIdx.x ] + SharedMem[ threadIdx.x - stride ];
		}

		__syncthreads( );

		if( threadIdx.x >= stride ){

			SharedMem[ threadIdx.x ] = temp;
		}
	}

	if( i < input_size ){

		output[ i ] = SharedMem[ threadIdx.x ];
	}
}

void kernel_setup( double*& host_input, double*& host_output, unsigned int input_size ){

	double* dev_input{ nullptr }, *dev_output{ nullptr };

	unsigned int allocation = input_size * sizeof( double );

	cudaMalloc( ( void** ) &dev_input, allocation );
	cudaMalloc( ( void** ) &dev_output, allocation );

	cudaMemcpy( dev_input, host_input, allocation, cudaMemcpyHostToDevice );

	dim3 blocks{ 1, 1, 1 };
	dim3 threads{ input_size };

	unsigned int shared_memsize = input_size * sizeof( double );

	Kogge_Stone_Exclusive_Kernel<<<blocks, threads, shared_memsize>>>( dev_input, dev_output, input_size );

	cudaMemcpy( host_output, dev_output, allocation, cudaMemcpyDeviceToHost );

	cudaFree( dev_input );
	cudaFree( dev_output );
}

int main( ){

	int size{ 0 };

	std::cout << "Enter the size of the array: ";
	std::cin >> size;

	double* array = generate_array( size );
	double* output = new double[ size ];

	std::cout << "\n";

	print_array( array, size );

	std::cout << "\n";

	kernel_setup( array, output, size );

	print_array( output, size );

	delete[ ] array, output;
}