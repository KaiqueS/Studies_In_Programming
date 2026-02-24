#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <algorithm>
#include <cmath>

/// TODO: fix the definition of ALL DESTRUCTORS! Use loops to correcltly delete the 2D-Arrays

struct PairOfArrays{

	PairOfArrays( ){ }
	PairOfArrays( PairOfArrays&& source ) : column( source.column ), values( source.values ), row_sizes( source.row_sizes ),
											maximum_rowsize( source.maximum_rowsize ), num_of_rows( source.num_of_rows ){

		source.column = nullptr;
		source.values = nullptr;
		source.row_sizes = nullptr;

		source.maximum_rowsize = 0;
		source.num_of_rows = 0;
	}
	~PairOfArrays( ){
		
		for( auto i = 0; i < num_of_rows; ++i ){

			delete[ ] column[ i ];
			delete[ ] values[ i ];
		}

		delete[ ] column;
		delete[ ] values;
		delete[ ] row_sizes;
	}

	int** column{ nullptr };
	int** values{ nullptr };

	int* row_sizes{ nullptr };

	int maximum_rowsize{ 0 };
	int num_of_rows{ 0 };
};


class COO{

public:

	COO( ){ }
	~COO( ){ delete[ ] rowIdx, colIdx, value; }

	void allocate_COO( int**& matrix, int rowSize, int colSize );
	void build_COO( int**& matrix, int rowSize, int colSize );
	void insert_element( int row, int col, int element );
	void reorder( int left_index, int right_index );

	__host__ __device__ int count_Zeroes( int**& matrix, int rowSize, int colSize );

	__host__ __device__ int*& get_rowIdx( ) { return rowIdx; }
	__host__ __device__ int*& get_colIdx( ) { return colIdx; }
	__host__ __device__ int*& get_value( ) { return value; }
	__host__ __device__ int get_size( ) const{ return size; }

private:

	int* rowIdx{ nullptr };
	int* colIdx{ nullptr };
	int* value{ nullptr };

	int size{ 0 };
	int zeroes{ 0 };
};

class CSR{

public:

	CSR( ){ }
	~CSR( ){ delete[ ] rowPtrs, colIdx, value; }

	void allocate_CSR( int**& matrix, int rowSize, int colSize );
	void build_CSR( int**& matrix, int rowSize, int colSize );

	__host__ __device__ int count_Zeroes( int**& matrix, int rowSize, int colSize );

	__host__ __device__ int*& get_rowPtrs( ) { return rowPtrs; }
	__host__ __device__ int*& get_colIdx( ) { return colIdx; }
	__host__ __device__ int*& get_value( ) { return value; }
	__host__ __device__ int& get_size( ) { return size; }

private:

	int* rowPtrs{ nullptr };
	int* colIdx{ nullptr };
	int* value{ nullptr };

	int size{ 0 };
	int zeroes{ 0 };
};

class ELL{

public:

	ELL( ){ }
	~ELL( ){ delete[ ] colIdx, value; }

	void build_ELL( int**& matrix, int rowsize, int colsize );

	PairOfArrays nonzero_matrix( int**& matrix, int rowsize, int colsize );
	PairOfArrays padded_matrix( int**& matrix, int rowsize, int colsize );

	const int* get_colIdx( ) const{ return colIdx; }
	const int* get_value( ) const{ return value; }

	const int get_size( ) const{ return size; }

private:

	int* colIdx{ nullptr };
	int* value{ nullptr };

	int size{ 0 };
};

class JDS{

public:

	JDS( ){ }
	~JDS( ){ delete[ ] iterPtr, colIdx, value, row; }

	PairOfArrays nonzero_matrix( int**& matrix, int rowsize, int colsize );

	const int* get_value( ) const{ return value; }

	const int get_size( ) const{ return size; }

	void sort_rows( PairOfArrays& matrix, int rowsize );
	void build_matrix( int**& matrix, int rowsize, int colsize );
	void build_row( int rowsize );

private:

	int* iterPtr{ nullptr };
	int* colIdx{ nullptr };
	int* value{ nullptr };

	int* row{ nullptr };

	int size{ 0 };
};

class ELL_COO : public ELL{

public:

	ELL_COO( ){ }
	~ELL_COO( );

	__host__ __device__ int**& get_ellcol( ){ return ell_colIdx; }
	__host__ __device__ int**& get_ellval( ){ return ell_value; }

	__host__ __device__ int*& get_coorow( ){ return coo_rowIdx; }
	__host__ __device__ int*& get_coocol( ){ return coo_colIdx; }
	__host__ __device__ int*& get_cooval( ){ return coo_value; }

	__host__ __device__ int& get_ell_rowsize( ){ return ell_rowsize; }
	__host__ __device__ int& get_ell_colsize( ){ return ell_colsize; }

	__host__ __device__ int& get_coo_size( ){ return coo_size; }
	__host__ __device__ int& get_coo_zeroes( ){ return coo_zeroes; }

	void build_ELL_COO( int**& matrix, int rowsize, int colsize );
	//void build_padding( int**& matrix, int rowsize, int colsize );

	//virtual PairOfArrays padded_matrix( int**& matrix, int rowsize, int colsize );

private:

	int** ell_colIdx{ nullptr };
	int** ell_value{ nullptr };

	int* coo_rowIdx{ nullptr };
	int* coo_colIdx{ nullptr };
	int* coo_value{ nullptr };

	int ell_rowsize{ 0 };
	int ell_colsize{ 0 };

	int coo_size{ 0 };
	int coo_zeroes{ 0 };
};