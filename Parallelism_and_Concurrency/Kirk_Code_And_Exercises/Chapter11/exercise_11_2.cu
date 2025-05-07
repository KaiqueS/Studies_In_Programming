#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <math.h>
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

__global__ void Kogge_Stone_scan_kernel( double* input, double* output, unsigned int N, int offset ){

	__shared__ extern double XY[ ]; // XY must have SECTION_SIZE elements, where SECTION_SIZE = Block size, i.e., amount of threads
	__shared__ extern double buffer[ ];
	
	unsigned int i = ( blockIdx.x * blockDim.x ) + threadIdx.x;

	XY[ threadIdx.x ] = ( i < N ) ? input[ i ] : 0.0;

	for( int stride = 1; stride < blockDim.x; stride *= 2 ){

		__syncthreads( );

		if( ( static_cast<int>( log2f( stride ) ) % 2 == 0 ) ){

			if( threadIdx.x >= stride ){

				buffer[ threadIdx.x + offset ] = XY[ threadIdx.x ] + XY[ threadIdx.x - stride ];
			}
		}

		else{

			if( threadIdx.x >= stride ){

				XY[ threadIdx.x ] = buffer[ threadIdx.x + offset ] + buffer[ threadIdx.x - stride + offset ];
			}
		}
	}

	if( i < N ){

		output[ i ] = XY[ threadIdx.x ];
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

	dim3 blocks{ 1 };
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