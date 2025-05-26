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
	//		 The rule below captures all cases and correctly rounds the quotient. The idea is - if rounding the quotient up results in an even ceiling, keep it, else, return
	//		 the floor of the quotient. -> WRONG. Check the correct index below
	/*if( Section_Size < 4 ){

		correct_rounding = static_cast<int>( ceil( static_cast<double>( Section_Size ) / 4.0 ) );
	}

	else{

		correct_rounding = ( static_cast<int>( ceil( static_cast<double>( Section_Size ) / 4.0 ) ) % 2 == 0 ) ? static_cast<int>( ceil( static_cast<double>( Section_Size ) / 4.0 ) ) :
																												static_cast<int>( floor( static_cast<double>( Section_Size ) / 4.0 ) );
	}*/

	// EXPLANATION: stride must be in the form of Powers of 2, but the code on the book allows for values that are not powers of 2. Thus, we pick the smallest power of 2 that is
	//				greater than the quotient Section_Size / 4.
	correct_rounding = static_cast<int>( pow( 2.0, ceil( log2( static_cast<double>( Section_Size ) / 4.0 ) ) ) );

	for( int stride = correct_rounding; stride > 0; stride /= 2 ){

		__syncthreads( );

		unsigned int index = ( ( threadIdx.x + 1 ) * stride * 2 ) - 1;

		// TEST: for input_size = 24, no matter threadIdx and stride, index + stride NEVER hits 23. Thus, the last element in Shared_Input is never hit
		if( ( index + stride ) < Section_Size ){

			Shared_Input[ index + stride ] += Shared_Input[ index ];
		}
	}

	__syncthreads( );

	if( i < input_size ){

		double holder = Shared_Input[ threadIdx.x ];

		output[ i ] = holder;
	}

	if( ( i + blockDim.x ) < input_size ){

		double holder = Shared_Input[ threadIdx.x + blockDim.x ];

		output[ i + blockDim.x ] = holder;
	}

	__syncthreads( );

	if( threadIdx.x == blockDim.x - 1 ){

		// NOTE: remember that Shared_Input is in shared-memory, which is BLOCK-LOCAL! I.e., e.g., blocks 1 and 2 do NOT share shared-memory address spaces!
		S[ blockIdx.x ] = Shared_Input[ Section_Size - 1 ];
	}
}

// NOTE: Shared Memory PERSISTS BETWEEN KERNEL CALLS! Keep that in mind, because, unless explicitly modified, values from previous kernel calls might still be
//		 in Shared Memory when you run other kernels that use it. But, since Shared Memory is block-local, I do not know what addresses intervals will be assigned
//		 to blocks from different kernels.
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

// NOTE: an interesting problem happened here when testing the code - To check if output was correct, I let the first GLOBAL thread, taken from 2 different blocks, print all output elements.
//		 The thing is, output is not modified by Second_Kernel, only by the First. This means that the printed values should match the values output has at the end of the First_Kernel call.
//		 But they did NOT match. Why? Because, the first GLOBAL thread, which is in the first block, printed values only AFTER the threads from the SECOND block modified output, which lies in
//		 global memory.
__global__ void Third_Kernel( double* S, double* output ){

	unsigned int i = ( blockIdx.x * blockDim.x ) + threadIdx.x;

	// EXPLANATION: all the partial prefix sums from the first section, which was assigned to the first block, are already correct. Thus, there is no need to
	//				add to them any value from S.
	if( blockIdx.x > 0 ){

		output[ i ] += S[ blockIdx.x - 1 ];
	}
}

void kernel_setup( double*& host_input, double*& host_output, unsigned int size ){

	double* dev_in{ nullptr }, *dev_out{ nullptr }, *dev_S{ nullptr };

	unsigned int array_size = size * sizeof( double );

	unsigned int num_of_blocks{ 0 };

	std::cout << "\nEnter the amount of blocks: ";
	std::cin >> num_of_blocks;

	cudaMalloc( ( void** ) &dev_in, array_size );
	cudaMalloc( ( void** ) &dev_out, array_size );
	cudaMalloc( ( void** ) &dev_S, ( num_of_blocks * sizeof( double ) ) );

	cudaMemcpy( dev_in, host_input, array_size, cudaMemcpyHostToDevice );

	dim3 blocks{ num_of_blocks, 1, 1 };
	dim3 threads{ static_cast<unsigned int>( std::ceil( ( static_cast<double>( size ) / ( 2.0 * static_cast<double>( num_of_blocks ) ) ) ) ) };

	unsigned int Section_Size = static_cast<unsigned int>( std::ceil( static_cast<double>( size / num_of_blocks ) ) );

	First_Kernel<<<blocks, threads, Section_Size>>>( dev_in, dev_out, dev_S, size, Section_Size );

	Second_Kernel<<<1, blocks, num_of_blocks>>>( dev_S, num_of_blocks, num_of_blocks );

	Third_Kernel<<<blocks, Section_Size>>>( dev_S, dev_out );

	cudaMemcpy( host_output, dev_out, array_size, cudaMemcpyDeviceToHost );

	cudaFree( dev_in );
	cudaFree( dev_out );
	cudaFree( dev_S );
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