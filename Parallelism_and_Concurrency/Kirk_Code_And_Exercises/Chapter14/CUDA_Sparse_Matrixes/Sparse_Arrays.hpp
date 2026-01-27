#pragma once

#include <algorithm>
#include <iterator>

struct PairOfArrays{

	PairOfArrays( ){ }
	~PairOfArrays( ){ delete[ ] column, values, row_sizes; }

	int** column{ nullptr };
	int** values{ nullptr };

	int* row_sizes{ nullptr };
};

class COO{

public:

	COO( ){ }
	~COO( ){ delete[ ] rowIdx, colIdx, value; }

	void allocate_COO( int**&matrix, int rowSize, int colSize );
	void build_COO( int**& matrix, int rowSize, int colSize );
	void insert_element( int row, int col, int element );
	void reorder( int left_index, int right_index );

	int count_Zeroes( int**& matrix, int rowSize, int colSize );

	const int* get_rowIdx( ) const{ return rowIdx; }
	const int* get_colIdx( ) const{ return colIdx; }
	const int* get_value( ) const{ return value; }

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

	void allocate_CSR( int**&matrix, int rowSize, int colSize );
	void build_CSR( int**& matrix, int rowSize, int colSize );

	int count_Zeroes( int**& matrix, int rowSize, int colSize );

	const int* get_rowPtrs( ) const{ return rowPtrs; }
	const int* get_colIdx( ) const{ return colIdx; }
	const int* get_value( ) const{ return value; }

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

private:

	int* colIdx{ nullptr };
	int* value{ nullptr };
};

class JDS{

public:
	
	JDS( ){ }
	~JDS( ){ delete[ ] iterPtr, colIdx, value, row; }

	int** nonzero_matrix( int**& matrix, int rowsize, int colsize );
	int** nonzero_colidx( int**& matrix, int rowsize, int colsize );

	const int* get_value( ) const{ return value; }

	void sort_rows( int**& matrix, int rowsize, int colsize );
	void build_matrix( int**& matrix, int rowsize, int colsize );
	void build_row( int**& matrix, int rowsize, int colsize );

private:

	int* iterPtr{ nullptr };
	int* colIdx{ nullptr };
	int* value{ nullptr };

	int* row{ nullptr };
};