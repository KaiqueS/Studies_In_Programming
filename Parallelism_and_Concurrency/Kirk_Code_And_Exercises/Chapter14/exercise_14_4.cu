
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

__global__ void SpMV_ELL_COO( ELL_COO* input, int* output ){


}

void kernel_setup( ELL_COO& host_input, int*& host_output, int rowsize, int colsize ){

	ELL_COO* dev_input{ };
	
	int* dev_output{ nullptr };

	cudaMalloc( ( void** ) &dev_input, sizeof( ELL_COO ) );
	cudaMalloc( ( void** ) &dev_output, colsize * sizeof( int ) );
	
	cudaMemcpy( dev_input, &host_input, sizeof( ELL_COO ), cudaMemcpyHostToDevice );

	// Copying data into dev_input member's pointers - START
	int** ell_colidx{ nullptr }, **ell_value{ nullptr };
	
	cudaMalloc( ( void** ) &ell_colidx, host_input.get_ell_rowsize( ) * sizeof( int* ) );
	cudaMalloc( ( void** ) &ell_value, host_input.get_ell_rowsize( ) * sizeof( int* ) );

	cudaMemcpy( ell_colidx, host_input.get_ellcol( ), host_input.get_ell_rowsize( ) * sizeof( int* ), cudaMemcpyHostToDevice );
	cudaMemcpy( ell_value, host_input.get_ellval( ), host_input.get_ell_rowsize( ) * sizeof( int* ), cudaMemcpyHostToDevice );

	// NOTE: this might be unecessary
	for( auto i = 0; i < host_input.get_ell_rowsize( ); ++i ){

		cudaMalloc( ( void** ) &ell_colidx[ i ], host_input.get_ell_colsize( ) * sizeof( int ) );
		cudaMalloc( ( void** ) &ell_value[ i ], host_input.get_ell_colsize( ) * sizeof( int ) );

		cudaMemcpy( ell_colidx[ i ], host_input.get_ellcol( )[ i ], host_input.get_ell_colsize( ) * sizeof( int ), cudaMemcpyHostToDevice );
		cudaMemcpy( ell_value[ i ], host_input.get_ellval( )[ i ], host_input.get_ell_rowsize( ) * sizeof( int ), cudaMemcpyHostToDevice );
	}

	cudaMemcpy( &( dev_input -> get_ellcol( ) ), &ell_colidx, sizeof( int* ), cudaMemcpyHostToDevice );
	cudaMemcpy( &( dev_input -> get_ellval( ) ), &ell_value, sizeof( int* ), cudaMemcpyHostToDevice );

	int* coo_row{ nullptr }, *coo_col{ nullptr }, *coo_value{ nullptr };

	cudaMalloc( ( void** ) &coo_row, host_input.get_coo_size( ) * sizeof( int ) );
	cudaMalloc( ( void** ) &coo_col, host_input.get_coo_size( ) * sizeof( int ) );
	cudaMalloc( ( void** ) &coo_value, host_input.get_coo_size( ) * sizeof( int ) );

	cudaMemcpy( coo_row, host_input.get_coorow( ), host_input.get_coo_size( ) * sizeof( int ), cudaMemcpyHostToDevice );
	cudaMemcpy( coo_col, host_input.get_coocol( ), host_input.get_coo_size( ) * sizeof( int ), cudaMemcpyHostToDevice );
	cudaMemcpy( coo_value, host_input.get_cooval( ), host_input.get_coo_size( ) * sizeof( int ), cudaMemcpyHostToDevice );

	cudaMemcpy( &( dev_input -> get_coorow( ) ), &host_input.get_coorow( ), sizeof( int* ), cudaMemcpyHostToDevice );
	cudaMemcpy( &( dev_input -> get_coocol( ) ), &host_input.get_coocol( ), sizeof( int* ), cudaMemcpyHostToDevice );
	cudaMemcpy( &( dev_input -> get_cooval( ) ), &host_input.get_cooval( ), sizeof( int* ), cudaMemcpyHostToDevice );

	// Copying data into dev_input member's pointers - END
}

int main( ){

	int rowsize{ 0 }, colsize{ 0 };

	std::cout << "Enter Row and Column sizes:";
	std::cin >> rowsize >> colsize;

	int** matrix = generate_matrix( rowsize, colsize );

	print_matrix( matrix, rowsize, colsize );

	ELL_COO ell_coo{ };
	ell_coo.build_ELL_COO( matrix, rowsize, colsize );

	int* output = new int[ colsize ]{ 0 };

	for( auto i = 0; i < rowsize; ++i ){

		delete[ ] matrix[ i ];
	}

	delete[ ] matrix;

	delete[ ] output;
}