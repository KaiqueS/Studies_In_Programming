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

// NOTE: I will be using the Brent-Kung Algorithm here.
// PROBLEM: when the size of the input array = 6, the last prefix sum is wrong
//			Seems like a rounding error when dividing Section_Size / 4
__global__ void First_Kernel( double* input, double* output, unsigned int input_size, unsigned int Section_Size ){

	__shared__ extern double Shared_Input[ ];

	unsigned int i = ( 2 * blockIdx.x * blockDim.x ) + threadIdx.x;

	if( i < input_size ){

		Shared_Input[ threadIdx.x ] = input[ i ];
		//Shared_Input[ threadIdx.x + ( input_size / 2 ) ] = input[ i + ( input_size / 2 ) ];
	}

	if( ( i + blockDim.x ) < input_size ){

		Shared_Input[ threadIdx.x + blockDim.x ] = input[ i + blockDim.x ];
	}

	for( unsigned int stride = 1; stride <= blockDim.x; stride *= 2 ){

		__syncthreads( );

		unsigned int index = ( ( threadIdx.x + 1 ) * ( 2 * stride ) ) - 1;

		if( index < Section_Size ){

			Shared_Input[ index ] += Shared_Input[ index -  stride ];
		}
	}

	int correct_rounding{ 0 };

	// NOTE: this is ugly, but necessary. The example code on the book is wrong. The code output is wrong/right depending on how Section_Size / 4 is rounded by the compiler.
	//		 The rule below captures all cases and correctly rounds the quocient.
	if( Section_Size < 4 ){

		correct_rounding = static_cast<int>( ceil( static_cast<double>( Section_Size ) / 4.0 ) );
	}

	else{

		correct_rounding = ( static_cast<int>( ceil( static_cast<double>( Section_Size ) / 4.0 ) ) % 2 == 0 ) ? static_cast<int>( ceil( static_cast<double>( Section_Size ) / 4.0 ) ) :
																												static_cast<int>( floor( static_cast<double>( Section_Size ) / 4.0 ) );
	}

	for( int stride = correct_rounding; stride > 0; stride /= 2 ){

		__syncthreads( );

		unsigned int index = ( ( threadIdx.x + 1 ) * stride * 2 ) - 1;

		if( ( index + stride ) < Section_Size ){

			Shared_Input[ index + stride ] += Shared_Input[ index ];
		}
	}

	__syncthreads( );

	if( i < input_size ){

		output[ i ] = Shared_Input[ threadIdx.x ];
	}

	if( ( i + blockDim.x ) < input_size ){

		output[ i + blockDim.x ] = Shared_Input[ threadIdx.x + blockDim.x ];
		//output[ i + blockDim.x + ( input_size / 2 ) ] = Shared_Input[ threadIdx.x + blockDim.x + ( input_size / 2 ) ];
	}
}

__global__ void Second_Kernel( ){


}

__global__ void Third_Kernel( ){


}

void kernel_setup( double*& host_input, double*& host_output, unsigned int size ){

	double* dev_in{ nullptr }, *dev_out{ nullptr };

	unsigned int array_size = size * sizeof( double );

	cudaMalloc( ( void** ) &dev_in, array_size );
	cudaMalloc( ( void** ) &dev_out, array_size );

	cudaMemcpy( dev_in, host_input, array_size, cudaMemcpyHostToDevice );

	dim3 blocks{ 1, 1, 1 };
	dim3 threads{ static_cast<unsigned int>( std::ceil( ( static_cast<double>( size ) / 2.0 ) ) ) };

	First_Kernel<<<blocks, threads, array_size>>>( dev_in, dev_out, size, size );

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

	std::cout << "\n";

	print_array( output, size );

	delete[ ] array, output;
}