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

/// PROBLEM: 8. Complete the host code and all three kernels for the segmented parallel scan algorithm in Fig. 11.9.

/// ANSWER:

// NOTE: I am using the Brent-Kung Algorithm here.
// PROBLEM: when the size of the input array = 6, the last prefix sum is wrong
//			Seems like a rounding error when dividing Section_Size / 4
// SOLVED: check EXPLANATION
__global__ void First_Kernel( double* input, double* output, double* S, unsigned int input_size, unsigned int Section_Size ){

	__shared__ extern double Shared_Input[ ];

	unsigned int i = ( 2 * blockIdx.x * blockDim.x ) + threadIdx.x;

	if( i < input_size ){

		Shared_Input[ threadIdx.x ] = input[ i ];
	}

	if( ( i + blockDim.x ) < input_size ){

		Shared_Input[ threadIdx.x + blockDim.x ] = input[ i + blockDim.x ];
	}

	// This is wrong. Somehow, we are leaving the for loop without the correct complete sum on the last element of shared_input
	for( unsigned int stride = 1; stride <= blockDim.x; stride *= 2 ){

		__syncthreads( );

		unsigned int index = ( ( threadIdx.x + 1 ) * 2 * stride ) - 1;

		if( index < Section_Size ){

			Shared_Input[ index ] += Shared_Input[ index - stride ];
		}
	}

	int correct_rounding{ 0 };

	// NOTE: this is ugly, but necessary. The example code on the book is wrong. The code output is wrong/right depending on how Section_Size / 4 is rounded by the compiler.
	//		 The rule below captures all cases and correctly rounds the quocient. The idea is - if rounding the quocient up results in an even ceiling, keep it, else, return
	//		 the floor of the quocient. -> WRONG. Check the correct index below
	/*if( Section_Size < 4 ){

		correct_rounding = static_cast<int>( ceil( static_cast<double>( Section_Size ) / 4.0 ) );
	}

	else{

		correct_rounding = ( static_cast<int>( ceil( static_cast<double>( Section_Size ) / 4.0 ) ) % 2 == 0 ) ? static_cast<int>( ceil( static_cast<double>( Section_Size ) / 4.0 ) ) :
																												static_cast<int>( floor( static_cast<double>( Section_Size ) / 4.0 ) );
	}*/

	// EXPLANATION: stride must be in the form of Powers of 2, but the code on the book allows for values that are not powers of 2. Thus, we pick the smallest power of 2 that is
	//				greater than the quocient Section_Size / 4.
	correct_rounding = static_cast<int>( pow( 2.0, ceil( log2( static_cast<double>( Section_Size ) / 4.0 ) ) ) );

	for( int stride = correct_rounding; stride > 0; stride /= 2 ){

		__syncthreads( );

		unsigned int index = ( ( threadIdx.x + 1 ) * stride * 2 ) - 1;

		// TEST: for input_size = 24, no matter threadIdx and stride, index + stride NEVER hits 23. Thus, the last element in Shared_Input is never hit
		if( ( index + stride ) < Section_Size ){

			if( index == 5 ){

				printf( "\n%d %d %d %f %f", threadIdx.x, ( index + stride ), stride, Shared_Input[ index + stride ], Shared_Input[ index ] );
			}

			Shared_Input[ index + stride ] += Shared_Input[ index ];
		}
	}

	__syncthreads( );

	if( i < input_size ){

		output[ i ] = Shared_Input[ threadIdx.x ];
	}

	if( ( i + blockDim.x ) < input_size ){

		output[ i + blockDim.x ] = Shared_Input[ threadIdx.x + blockDim.x ];
	}

	__syncthreads( );

	S[ blockIdx.x ] = ( threadIdx.x == ( blockDim.x - 1 ) ) ? Shared_Input[ Section_Size - 1 ] : 0.0;
}

__global__ void Second_Kernel( double* S, unsigned int input_size, unsigned int Section_Size ){

	__shared__ extern double Shared_Input[ ];

	unsigned int i = ( 2 * blockIdx.x * blockDim.x ) + threadIdx.x;

	if( i < input_size ){

		Shared_Input[ threadIdx.x ] = S[ i ];
	}

	if( ( i + blockDim.x ) < input_size ){

		Shared_Input[ threadIdx.x + blockDim.x ] = S[ i + blockDim.x ];
	}

	for( unsigned int stride = 1; stride <= blockDim.x; stride *= 2 ){

		__syncthreads( );

		unsigned int index = ( ( threadIdx.x + 1 ) * ( 2 * stride ) ) - 1;

		if( index < Section_Size ){

			Shared_Input[ index ] += Shared_Input[ index -  stride ];
		}
	}

	int correct_rounding{ 0 };

	correct_rounding = static_cast<int>( pow( 2.0, ceil( log2( static_cast<double>( Section_Size ) / 4.0 ) ) ) );

	for( int stride = correct_rounding; stride > 0; stride /= 2 ){

		__syncthreads( );

		unsigned int index = ( ( threadIdx.x + 1 ) * stride * 2 ) - 1;

		if( ( index + stride ) < Section_Size ){

			Shared_Input[ index + stride ] += Shared_Input[ index ];
		}
	}

	__syncthreads( );

	if( i < input_size ){

		S[ i ] = Shared_Input[ threadIdx.x ];
	}

	if( ( i + blockDim.x ) < input_size ){

		S[ i + blockDim.x ] = Shared_Input[ threadIdx.x + blockDim.x ];
	}
}

__global__ void Third_Kernel( double* S, double* output ){

	unsigned int i = ( blockIdx.x * blockDim.x ) + threadIdx.x;

	output[ i ] += S[ blockIdx.x - 1 ];
}

void kernel_setup( double*& host_input, double*& host_output, unsigned int size ){

	double* dev_in{ nullptr }, *dev_out{ nullptr }, *dev_S{ nullptr };

	unsigned int array_size = size * sizeof( double );

	cudaMalloc( ( void** ) &dev_in, array_size );
	cudaMalloc( ( void** ) &dev_out, array_size );

	cudaMemcpy( dev_in, host_input, array_size, cudaMemcpyHostToDevice );

	unsigned int num_of_blocks{ 0 };

	std::cout << "\nEnter the amount of blocks: ";
	std::cin >> num_of_blocks;

	dim3 blocks{ num_of_blocks, 1, 1 };
	dim3 threads{ static_cast<unsigned int>( std::ceil( ( static_cast<double>( size ) / 2.0 ) ) ) };

	unsigned int Section_Size = static_cast<unsigned int>( std::ceil( static_cast<double>( size / num_of_blocks ) ) );

	cudaMalloc( ( void** ) &dev_S, ( num_of_blocks * sizeof( double ) ) );

	First_Kernel<<<blocks, threads, Section_Size>>>( dev_in, dev_out, dev_S, size, Section_Size );

	cudaMemcpy( host_output, dev_out, array_size, cudaMemcpyDeviceToHost );

	cudaFree( dev_in );
	cudaFree( dev_out );
}

int main( ){

	unsigned int size{ 0 };

	std::cout << "Enter the size of the array: ";
	std::cin >> size;

	std::cout << "\n";

	double* array = generate_array( size );
	double* output = new double[ size ];

	print_array( array, size );

	kernel_setup( array, output, size );

	std::cout << "\n\n";

	print_array( output, size );

	// TODO: find Shared Memory maximum size. Use it to determine the amount of blocks needed

	delete[ ] array, output;
}