#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>

// NOTE: maybe I should have used std::vector or unique_ptr, instead of raw arrays?
class COO{

public:

	COO( ){ }
	~COO( ){ rowIdx.clear( ); colIdx.clear( ); value.clear( ); }

	void allocate_COO( int**&matrix, int rowSize, int colSize );
	void build_COO( int**& matrix, int rowSize, int colSize );
	void insert_element( int row, int col, int element );

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


private:


};

class ELL{

public:


private:


};

class JDS{

public:


private:


};