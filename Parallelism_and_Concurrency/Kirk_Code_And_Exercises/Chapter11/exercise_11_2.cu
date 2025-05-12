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

void print_array( double* array, int size ){

	for( auto i = 0; i < size; ++i ){

		std::cout << array[ i ] << " ";
	}

	std::cout << "\n";
}

/// PROBLEM: 2. Modify the Kogge-Stone parallel scan kernel in Fig. 11.3 to use double-buffering instead of a second call to __syncthreads()
///				to overcome the write-after-read race condition.

/// ANSWER:

// PROBLEM: as soon as N > 9, partial sums get wrong precisely at the midpoint of the array
__global__ void Kogge_Stone_scan_kernel( double* input, double* output, unsigned int N, int offset ){

	__shared__ extern double XY[ ]; // XY must have SECTION_SIZE elements, where SECTION_SIZE = Block size, i.e., amount of threads
	__shared__ extern double buffer[ ];
	
	unsigned int i = ( blockIdx.x * blockDim.x ) + threadIdx.x;

	XY[ threadIdx.x ] = ( i < N ) ? input[ i ] : 0.0;

	for( int stride = 1; stride < blockDim.x; stride *= 2 ){

		__syncthreads( );

		// NOTE: static_cast to int from double was INCORRECTLY rounding DOWN the logarithm. To solve that, I had to add 0.5.
		if( ( static_cast<int>( log2( static_cast<double>( stride ) + 0.5 ) ) ) % 2 == 0 ){

			if( threadIdx.x >= stride ){

				buffer[ threadIdx.x + offset ] = XY[ threadIdx.x ] + XY[ threadIdx.x - stride ];
			}

			else{

				buffer[ threadIdx.x + offset ] = XY[ threadIdx.x ]; // NOTE: this is necessary. As iterations go on, threads with indexes SMALLER than stride will not update their required outcome elements
																	//		 without this ELSE clause. This, in turn, makes the partial sums for subsequent elements incorrect!
			}
		}

		else{

			if( threadIdx.x >= stride ){

				XY[ threadIdx.x ] = buffer[ threadIdx.x + offset ] + buffer[ threadIdx.x - stride + offset ];
			}

			else{

				XY[ threadIdx.x ] = buffer[ threadIdx.x + offset ]; // NOTE: this is necessary. As iterations go on, threads with indexes SMALLER than stride will not update their required outcome elements
																	//		 without this ELSE clause. This, in turn, makes the partial sums for subsequent elements incorrect!
			}
		}
	}

	if( i < N ){

		output[ i ] = ( static_cast<int>( ceil( log2( static_cast<double>( N ) ) ) ) % 2 == 0 ) ? XY[ threadIdx.x ] : buffer[ threadIdx.x + offset ];
	}
}

void kernel_launch( double*& host_input, double*& host_output, int size ){

	double* dev_input{ nullptr }, *dev_output{ nullptr };

	unsigned int array_size = size * sizeof( double );

	cudaMalloc( ( void** ) &dev_input, array_size );
	cudaMalloc( ( void** ) &dev_output, array_size );

	cudaMemcpy( dev_input, host_input, array_size, cudaMemcpyHostToDevice );
	
	int section_size{ 0 };

	std::cout << "\nEnter the section size: ";
	std::cin >> section_size;

	section_size *= sizeof( double );

	std::cout << "\n";

	unsigned int num_of_threads{ 0 };

	std::cout << "Enter the number of threads: ";
	std::cin >> num_of_threads;

	std::cout << "\n";

	dim3 blocks{ 1 }; // TODO: fix this! This is incorrect. The correct amount of blocks should be input size / section size
	dim3 threads{ num_of_threads };

	Kogge_Stone_scan_kernel<<<blocks, threads, section_size>>>( dev_input, dev_output, size, section_size );

	cudaMemcpy( host_output, dev_output, array_size, cudaMemcpyDeviceToHost );

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

	kernel_launch( array, output, size );

	print_array( output, size );

	delete[ ] array, output;
}