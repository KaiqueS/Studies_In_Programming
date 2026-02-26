
#include "/Projetos/Studies_In_Programming/Parallelism_and_Concurrency/Kirk_Code_And_Exercises/Chapter14/CUDA_Sparse_Matrixes/Sparse_Arrays.cpp"

#include <iostream>
#include <stdio.h>
#include <random>

int**& generate_matrix( int rowsize, int colsize ){

	int** matrix = new int*[ rowsize ]{ nullptr };

	std::random_device dev;
	std::uniform_int_distribution<int> dist( -( rowsize * colsize ), ( rowsize * colsize ) );
	std::mt19937_64 rng( dev( ) );

	for( auto i = 0; i < rowsize; ++i ){

		matrix[ i ] = new int[ colsize ]{ 0 };

		for( auto j = 0; j < colsize; ++j ){

			matrix[ i ][ j ] = dist( rng );
		}
	}

	return matrix;
}

void print_matrix( int**& matrix, int rowsize, int colsize ){

	for( auto i = 0; i < rowsize; ++i ){

		for( auto j = 0; j < colsize; ++j ){

			printf( "%d ", matrix[ i ][ j ] );
		}

		printf( "\n" );
	}
}

void print_matrix( int*& matrix, int size ){

	for( auto i = 0; i < size; ++i ){

		printf( "%d ", matrix[ i ] );
	}

	printf( "\n" );
}

/// PROBLEM 4. Implement the host code for producing the hybrid ELL-COO format and using it to perform SpMV.
///            Launch the ELL kernel to execute on the device, and compute the contributions of the COO elements on the host.

__global__ void SpMV_ELL_COO( ELL_COO* input, int** output ){


}

void kernel_setup( ELL_COO host_input ){


}

int main( ){

	int rowsize{ 0 }, colsize{ 0 };

	std::cout << "Enter Row and Column sizes:";
	std::cin >> rowsize >> colsize;

	int** matrix = generate_matrix( rowsize, colsize );

	print_matrix( matrix, rowsize, colsize );

	ELL_COO ell_coo;
	ell_coo.build_ELL_COO( matrix, rowsize, colsize );
}