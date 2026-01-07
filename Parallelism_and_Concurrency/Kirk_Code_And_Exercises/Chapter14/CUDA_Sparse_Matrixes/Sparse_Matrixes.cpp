#include "Sparse_Matrixes.hpp"

/// COO - BEGIN


// NOTE: I could parallelize this later using tiling to speed things up.
int COO::count_Zeroes( int**& matrix, int rowSize, int colSize ){

	int counter{ 0 };

	for( auto i = 0; i < rowSize; ++i ){

		for( auto j = 0; j < colSize; ++j ){

			if( matrix[ i ][ j ] == 0 ){

				++counter;
			}

			else{

				continue;
			}
		}
	}

	return counter;
}

// NOTE: COO arrays are unidimensional
void COO::allocate_COO( int**&matrix, int rowSize, int colSize ){

	int zeroes{ count_Zeroes( matrix, rowSize, colSize ) };

	rowIdx.reserve( ( rowSize * colSize ) - zeroes );
	colIdx.reserve( ( rowSize * colSize ) - zeroes );
	value.reserve( ( rowSize * colSize ) - zeroes );
}

void COO::build_COO( int**& matrix, int rowSize, int colSize ){

	allocate_COO( matrix, rowSize, colSize );

	for( auto i = 0; i < rowSize; ++i ){

		for( auto j = 0; j < colSize; ++j ){

			// NOTE: any time we hit a 0, we use zeroes to put matrix[ i ][ j ] into the correct position in value, rowIdx, and colIdx
			if( matrix[ i ][ j ] != 0 ){

				value.push_back( matrix[ i ][ j ] );
				rowIdx.push_back( i );
				colIdx.push_back( j );
			}
			
			// NOTE: If we hit a 0 in matrix, we increment the offset zeroes
			else{

				continue;
			}
		}
	}
}

void COO::insert_element( int row, int col, int element ){

	rowIdx.push_back( row );
	colIdx.push_back( col );
	value.push_back( element );
}

/// END - COO

// ------------------

/// CSR - BEGIN


/// END - CSR

// ------------------

/// ELL - BEGIN


/// END - ELL

// ------------------