#include <cuda.h>
#include <vector>
#include <iostream>
#include <random>

/// 1. A matrix addition takes two input matrices A and B and produces one output matrix C. Each element of the output matrix C is the sum of the corresponding
/// elements of the input matrices A and B, i.e., C[i][j] = A[i][j] + B[i][j]. For simplicity, we will only handle square matrices whose elements are
/// single-precision floating-point numbers. Write a matrix addition kernel and the host stub function that can be called with four parameters: pointer-
/// to-the-output matrix, pointer-to-the-first-input matrix, pointer-to-the-second-input matrix, and the number of elements in each dimension. Follow the
/// instructions below:

// B. Write a kernel that has each thread to produce one output matrix element. Fill in the execution configuration parameters for this design.

__global__
void matrixAddKernel_B( float* output, float* first_input, float* second_input, int size ){

    int matrix_range{ size * size };

    int thread_id = ( blockIdx.x * blockDim.x ) + threadIdx.x;

    if( thread_id < matrix_range ){

        output[ thread_id ] = first_input[ thread_id ] + second_input[ thread_id ];
    }
}

// C. Write a kernel that has each thread to produce one output matrix row. Fill in the execution configuration parameters for the design.

__global__
void matrixAddKernel_C( float* output, float* first_input, float* second_input, int size ){

    
}

// D. Write a kernel that has each thread to produce one output matrix column. Fill in the execution configuration parameters for the design.

__global__
void matrixAddKernel_D( float* output, float* first_input, float* second_input, int size ){

    
}

// E. Analyze the pros and cons of each kernel design above.

// A. Write the host stub function by allocating memory for the input and output matrices, transferring input data to device; launch the kernel, transferring the
// output data to host and freeing the device memory for the input and output data. Leave the execution configuration parameters open for this step.

void set_up( float* h_output, float* h_first_input, float* h_second_input, int dim ){

    float *d_Output, *d_first, *d_second;
    
    int size = dim * sizeof( float );

    cudaMalloc( ( void** ) &d_first, ( size * size ) );
    cudaMalloc( ( void** ) &d_second, ( size * size ) );
    cudaMalloc( ( void** ) &d_Output, ( size * size ) );

    cudaMemcpy( d_first, h_first_input, size, cudaMemcpyHostToDevice );
    cudaMemcpy( d_second, h_second_input, size, cudaMemcpyHostToDevice );

    matrixAddKernel_B<<<32,32>>>( d_Output, d_first, d_second, dim );

    cudaMemcpy( h_output, d_Output, size, cudaMemcpyDeviceToHost );

    cudaFree( d_Output );
    cudaFree( d_first );
    cudaFree( d_second );
}

int main( ){


}