#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>
#include <iostream>
#include <random>

void fill_matrix( float**& matrix, int size ){

    std::random_device dev;
    std::uniform_real_distribution<float> dist( -( size * size ), ( size * size ) );
    std::mt19937_64 rng( dev( ) );

    //int dim = size * size;

    matrix = new float*[ size ];

    for( auto i = 0; i < size; ++i ){

        matrix[ i ] = new float[ size ];

        for( auto j = 0; j < size; ++j ){

            matrix[ i ][ j ] = dist( rng );
        }
    }
}

void print_matrix( float**& matrix, int size ){

    for( auto i = 0; i < size; ++i ){

        for( auto j = 0; j < size; ++j ){

            printf( "%lf ", matrix[ i ][ j ] );

            if( j == size - 1 ){

                printf( "\n" );
            }
        }
    }
}

void print_matrix( float*& matrix, int size ){

    int double_dim = ( size * size );

    for( auto i = 0; i < double_dim; ++i ){

        printf( "%lf ", matrix[ i ] );

        if( ( i + 1 ) % size == 0 ){

            printf( "\n" );
        }
    }
}

float*& simple_flattening( float**& matrix, int dimension ){

    float* flat_mat = new float[ dimension * dimension ];

    for( auto i = 0; i < dimension; ++i ){

        for( auto j = 0; j < dimension; ++j ){

            flat_mat[ ( i * dimension ) + j ] = matrix[ i ][ j ];
        }
    }

    return flat_mat;
}

/// NOTE: Remember that C++ is NOT C. We cannot handle multidimensional arrays as we do in C, because in C++ there is NO
///       guarantee that, for T** Obj, Obj[ 0 ][ 0 ] and Obj[ 1 ][ 0 ] are contiguous in memory. I.e., they are not stored
///       sequentially in memory, even though Obj[ 0 ] and Obj[ 1 ] are. Thus, first we must flatten our arrays, since it
///       is not trivial to handle multidimensional arrays in CUDA.

/// 1. A matrix addition takes two input matrices A and B and produces one output matrix C. Each element of the output matrix C is the sum of the corresponding
/// elements of the input matrices A and B, i.e., C[i][j] = A[i][j] + B[i][j]. For simplicity, we will only handle square matrices whose elements are
/// single-precision floating-point numbers. Write a matrix addition kernel and the host stub function that can be called with four parameters: pointer-
/// to-the-output matrix, pointer-to-the-first-input matrix, pointer-to-the-second-input matrix, and the number of elements in each dimension. Follow the
/// instructions below:

/// B. Write a kernel that has each thread to produce one output matrix element. Fill in the execution configuration parameters for this design.

// Problem: how to generalize for instances where the dimensions of the matrix are greater than the available amount of threads?
// Solution: let M be a square matrix of dimensions N x N, and B be a square block of dimensions A x A, s.t. A < N. Then, there
//           are 0 < N² - A² elements uncovered by B. After B processes a A x A section of M, we just assign B to those N² - A²
//           remaining elements.
__global__ void matrixAddKernel_B( float* output, float* first_input, float* second_input, int size ){

    int row = ( blockIdx.x * blockDim.x ) + threadIdx.x;
    int col = ( blockIdx.y * blockDim.y ) + threadIdx.y;

    if( ( row < size ) && ( col < size ) ){

        output[ ( row * size ) + col ] = first_input[ ( row * size ) + col ] + second_input[ ( row * size ) + col ];
    }
}

// C. Write a kernel that has each thread to produce one output matrix row. Fill in the execution configuration parameters for the design.

__global__ void matrixAddKernel_C( float* output, float* first_input, float* second_input, int size ){

    int row = ( blockIdx.x * blockDim.x ) + threadIdx.x;

    if( row < size ){

        for( auto i = 0; i < size; ++i ){

            output[ ( row * size ) + i ] = first_input[ ( row * size ) + i ] + second_input[ ( row * size ) + i ];
        }
    }
}

// D. Write a kernel that has each thread to produce one output matrix column. Fill in the execution configuration parameters for the design.

__global__ void matrixAddKernel_D( float* output, float* first_input, float* second_input, int size ){

    int col = ( blockIdx.y * blockDim.y ) + threadIdx.y;

    if( col < size ){

        for( auto i = 0; i < size; ++i ){

            output[ ( col * size ) + i ] = first_input[ ( i * size ) + col ] + second_input[ ( i * size ) + col ];
        }
    }
}

// E. Analyze the pros and cons of each kernel design above.

// A. Write the host stub function by allocating memory for the input and output matrices, transferring input data to device; launch the kernel, transferring the
// output data to host and freeing the device memory for the input and output data. Leave the execution configuration parameters open for this step.

void set_up( float*& h_output, float**& h_first_input, float**& h_second_input, int dim ){

    float *d_Output{ nullptr }, *d_first{ nullptr }, *d_second{ nullptr };
    
    int size = ( dim * dim ) * sizeof( float ); // NOTE: DO NOT FORGET sizeof(), because allocation is made in terms of BYTES!

    cudaMalloc( ( void** ) &d_first, size );
    cudaMalloc( ( void** ) &d_second, size );
    cudaMalloc( ( void** ) &d_Output, size );

    float* flat_first = simple_flattening( h_first_input, dim );
    float* flat_second = simple_flattening( h_second_input, dim );

    cudaMemcpy( d_first, flat_first, size, cudaMemcpyHostToDevice );
    cudaMemcpy( d_second, flat_second, size, cudaMemcpyHostToDevice );

    int blocks{ 0 }, threads{ 0 };

    printf( "Enter the Grid and Block sizes: \n" );
    std::cin >> blocks >> threads;

    dim3 dimGrid( blocks, 1, 1 );
    dim3 dimBlock( threads,threads, 1 );

    //matrixAddKernel_B<<<dimGrid, dimBlock>>>( d_Output, d_first, d_second, dim );
    //matrixAddKernel_C<<<dimGrid, dimBlock>>>( d_Output, d_first, d_second, dim );
    matrixAddKernel_D<<<dimGrid, dimBlock>>>( d_Output, d_first, d_second, dim );

    h_output = new float[ size ];

    cudaMemcpy( h_output, d_Output, size, cudaMemcpyDeviceToHost );

    cudaFree( d_Output );
    cudaFree( d_first );
    cudaFree( d_second );
}

int main( ){

    float* output{ nullptr };
    float** first{ nullptr }, **second{ nullptr };

    int dimension{ 0 };

    std::cout << "Enter the matrix dimensions:" << "\n";
    std::cin >> dimension;

    fill_matrix( first, dimension );
    fill_matrix( second, dimension );

    printf( "\n" );

    print_matrix( first, dimension );

    printf( "\n" );

    print_matrix( second, dimension );

    printf( "\n" );

    set_up( output, first, second, dimension );

    print_matrix( output, dimension );

    std::cout << "\n";

    delete[ ] output;
    delete[ ] first, second;
}