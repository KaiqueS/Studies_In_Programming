#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <random>
#include <math.h>
#include <algorithm>

struct Pair{

	int first{ 0 };
	int second{ 0 };
};

int*& generate_array( int size ){

	int* array = new int[ size ];

	std::random_device dev;
	std::uniform_int_distribution<int> dist( -( size * size ), ( size * size ) );
	std::mt19937_64 rng( dev( ) );

	for( auto i = 0; i < size; ++i ){

		array[ i ] = dist( rng );
	}

	return array;
}

Pair split_array( int* input, int*& left, int*& right, int size ){

	Pair values{ };

	int split = ( size % 2 == 0 ) ? ( size / 2 ) : static_cast<int>( std::ceil( static_cast<double>( size ) / 2.0 ) );

	int remainder = size - split;

	values.first = split;
	values.second = remainder;

	left = new int[ split ];
	right = new int[ remainder ];

	for( auto i = 0; i < split; ++i ){

		left[ i ] = input[ i ];

		if( i < remainder ){

			right[ i ] = input[ split + i ];
		}

		else{

			continue;
		}
	}

	return values;
}

void print_array( int*& array, int size ){

	for( auto i = 0; i < size; ++i ){

		printf( "%d ", array[ i ] );
	}

	printf( "\n" );
}

// FOR TESTING ONLY. THIS IS INEFFICIENT
void linear_sort( int*& array, int size ){

	int holder{ 0 };

	for( auto i = 0; i < size; ++i ){

		for( auto j = i + 1; j < size; ++j ){

			if( array[ i ] > array[ j ] ){

				holder = array[ i ];

				array[ i ] = array[ j ];
				array[ j ] = holder;
			}

			else{

				continue;
			}
		}
	}
}

void merge_sequential( int* A, int m, int* B, int n, int*& C ){

	int i{ 0 }, j{ 0 }, k{ 0 };

	while( ( i < m ) && ( j < n ) ){

		C[ k++ ] = ( A[ i ] <= B[ j ] ) ? A[ i++ ] : B[ j++ ];
	}

	if( i == m ){

		while( j < n ){

			C[ k++ ] = B[ j++ ];
		}
	}

	else{

		while( i < m ){

			C[ k++ ] = A[ i++ ];
		}
	}
}

void co_rank( ){ }

__global__ void merge_basic_kernel( int* A, int m, int* B, int n, int* C ){


}

__global__ void merge_tiled_kernel( int* A, int m, int* B, int n, int* C, int tile_size ){


}

__global__ void merge_ciruclar_buffer_kernel( int* A, int m, int* B, int n, int* C ){


}


void kernel_setup( ){


}

int main( ){

	int size{ 0 };

	std::cout << "Enter the size of the array: ";
	std::cin >> size;

	int* array = generate_array( size );
	int* left{ nullptr }, *right{ nullptr };
	int* output = new int[ size ];

	Pair values = split_array( array, left, right, size );

	print_array( array, size );

	linear_sort( left, values.first );
	linear_sort( right, values.second );

	merge_sequential( left, values.first, right, values.second, output );

	print_array( output, size );

	delete[ ] array, left, right, output;
}