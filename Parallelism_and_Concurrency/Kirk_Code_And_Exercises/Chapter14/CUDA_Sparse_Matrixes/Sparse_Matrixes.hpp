#pragma once

//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

#include <vector>

class COO{

public:

	COO( ){ }
	~COO( ){ rowIdx.clear( ); colIdx.clear( ); value.clear( ); }

	void allocate_COO( int**&matrix, int rowSize, int colSize );
	void build_COO( int**& matrix, int rowSize, int colSize );
	void insert_element( int row, int col, int element );
	void reorder( int left_index, int right_index );

	int count_Zeroes( int**& matrix, int rowSize, int colSize );

	const std::vector<int>& get_rowIdx( ) const{ return rowIdx; }
	const std::vector<int>& get_colIdx( ) const{ return colIdx; }
	const std::vector<int>& get_value( ) const{ return value; }

private:

	std::vector<int> rowIdx{ };
	std::vector<int> colIdx{ };
	std::vector<int> value{ };
};

class CSR{

public:

	CSR( ){ }
	~CSR( ){ rowPtrs.clear( ); colIdx.clear( ); value.clear( ); }

	void allocate_CSR( int**&matrix, int rowSize, int colSize );
	void build_CSR( int**& matrix, int rowSize, int colSize );

	int count_Zeroes( int**& matrix, int rowSize, int colSize );

	const std::vector<int>& get_rowPtrs( ) const{ return rowPtrs; }
	const std::vector<int>& get_colIdx( ) const{ return colIdx; }
	const std::vector<int>& get_value( ) const{ return value; }

private:

	std::vector<int> rowPtrs{ };
	std::vector<int> colIdx{ };
	std::vector<int> value{ };
};

class ELL{

public:

	ELL( ){ }
	~ELL( ){ colIdx.clear( ); value.clear( ); }

	void build_ELL( int**& matrix, int rowsize, int colsize );

	std::vector<std::vector<int>> nonzero_matrix( int**& matrix, int rowsize, int colsize );
	std::vector<std::vector<int>> padded_matrix( int**& matrix, int rowsize, int colsize );

	const std::vector<int>& get_colIdx( ) const{ return colIdx; }
	const std::vector<int>& get_value( ) const{ return value; }

private:

	std::vector<int> colIdx{ };
	std::vector<int> value{ };
};

// TODO: I forgot to modify iterPtr to hold the offset between elements from the same row
class JDS{

public:
	
	JDS( ){ }
	~JDS( ){ iterPtr.clear( ); colIdx.clear( ); value.clear( ); }

	std::vector<std::vector<int>> nonzero_matrix( int**& matrix, int rowsize, int colsize );
	std::vector<std::vector<int>> nonzero_colidx( int**& matrix, int rowsize, int colsize );

	const std::vector<int>& get_value( ) const{ return value; }

	void sort_rows( std::vector<std::vector<int>>& matrix );
	void build_matrix( int**& matrix, int rowsize, int colsize );
	void build_row( std::vector<std::vector<int>>& matrix );

private:

	std::vector<int> iterPtr{ };
	std::vector<int> colIdx{ };
	std::vector<int> value{ };

	std::vector<int> row{ };
};