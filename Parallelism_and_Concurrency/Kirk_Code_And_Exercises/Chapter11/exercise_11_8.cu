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

__global__ void First_Kernel( ){

}

__global__ void Second_Kernel( ){


}

__global__ void Third_Kernel( ){


}

void kernel_setup( ){


}

int main( ){

	int size{ 0 };

	std::cout << "Enter the size of the array: ";
	std::cin >> size;

	std::cout << "\n";

	double* array = generate_array( size );

	print_array( array, size );

	delete[ ] array;
}