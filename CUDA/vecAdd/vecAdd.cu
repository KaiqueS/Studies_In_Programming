#include <cuda.h>
#include <vector>
#include <iostream>

__global__
void vecAddKernel( double* A, double* B, double* C, int n ){

	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if( i < n ){

		C[ i ] = A[ i ] + B[ i ];
	}
}

void vecAdd( double* h_A, double* h_B, double* h_C, int n ){

	double *d_A, *d_B, *d_C;

	int size = n * sizeof( double );
	
	cudaMalloc( ( void** )&d_A, size );
	cudaMemcpy( d_A, h_A, size, cudaMemcpyHostToDevice );

	cudaMalloc( ( void** )&d_B, size );
	cudaMemcpy( d_B, h_B, size, cudaMemcpyHostToDevice );
	
	cudaMalloc( ( void** )&d_C, size );

	vecAddKernel<<<ceil( n / 256.0 ), 256>>>( d_A, d_B, d_C, n );

	cudaMemcpy( h_C, d_C, size, cudaMemcpyDeviceToHost );

	for( auto i = 0; i < n; ++i ){

		std::cout << h_C[ i ] << " ";
	}

	std::cout << "\n";

	cudaFree( d_A );
	cudaFree( d_B );
	cudaFree( d_C );
}

int main( ){

	double A[ 5 ] = { 1.34, 2.565, 3.56, 4.123, 5.45 };
	double B[ 5 ] = { 1.78566, 2.24656, 3.2345, 4.456, 5.7567 };
	double C[ 5 ]{ };

	vecAdd( A, B, C, 5 );
}