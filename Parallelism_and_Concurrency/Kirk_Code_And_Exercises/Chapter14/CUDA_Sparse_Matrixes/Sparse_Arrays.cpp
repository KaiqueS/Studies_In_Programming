#include "Sparse_Arrays.hpp"

/// COO - BEGIN

// NOTE: I could parallelize this later using tiling to speed things up.
__host__ __device__  int COO::count_Zeroes( int**& matrix, int rowSize, int colSize ){

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

	zeroes = count_Zeroes( matrix, rowSize, colSize );

	rowIdx = new int[ ( rowSize * colSize ) - zeroes ];
	colIdx = new int[ ( rowSize * colSize ) - zeroes ];
	value = new int[ ( rowSize * colSize ) - zeroes ];

	size = ( rowSize * colSize ) - zeroes;
}

void COO::build_COO( int**& matrix, int rowSize, int colSize ){

	allocate_COO( matrix, rowSize, colSize );

	int counter{ 0 };

	for( auto i = 0; i < rowSize; ++i ){

		for( auto j = 0; j < colSize; ++j ){

			// NOTE: any time we hit a 0, we use zeroes to put matrix[ i ][ j ] into the correct position in value, rowIdx, and colIdx
			if( matrix[ i ][ j ] != 0 ){

				// NOTE: I could use the mapping ( ( i * rowsize ) + j - counter ) if I increment counter on the else block
				rowIdx[ counter ] = i;
				colIdx[ counter ] = j;
				value[ counter ] = matrix[ i ][ j ];

				++counter;
			}
			
			else{

				continue;
			}
		}
	}
}

// NOTE: needs testing
void COO::insert_element( int row, int col, int element ){

	int* newRow = new int[ size + 1 ];
	int* newCol = new int[ size + 1 ];
	int* newVal = new int[ size + 1 ];

	// NOTE: maybe ( size - 1 ) instead of size
	std::copy( &rowIdx[ 0 ], &rowIdx[ 0 ] + size, &newRow[ 0 ] );
	std::copy( &colIdx[ 0 ], &colIdx[ 0 ] + size, &newCol[ 0 ] );
	std::copy( &value[ 0 ], &value[ 0 ] + size, &newVal[ 0 ] );

	newRow[ size ] = row;
	newCol[ size ] = col;
	newVal[ size ] = element;

	++size;

	rowIdx = std::move( newRow );
	colIdx = std::move( newCol );
	value = std::move( newVal );

	delete[ ] newRow, newCol, newVal;
}

// NOTE: needs testing
void COO::reorder( int left_index, int right_index ){

	std::swap( rowIdx[ left_index ], rowIdx[ right_index ] );
	std::swap( colIdx[ left_index ], colIdx[ right_index ] );
	std::swap( value[ left_index ], value[ right_index ] );
}

/// END - COO

// ------------------

/// CSR - BEGIN

__host__ __device__ int CSR::count_Zeroes( int**& matrix, int rowSize, int colSize ){

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

// NOTE: CSR arrays are unidimensional
void CSR::allocate_CSR( int**&matrix, int rowSize, int colSize ){

	zeroes = count_Zeroes( matrix, rowSize, colSize );

	rowPtrs = new int[ rowSize + 1 ]{ 0 };
	colIdx = new int[ ( rowSize * colSize ) - zeroes ];
	value = new int[ ( rowSize * colSize ) - zeroes ];

	size = ( rowSize * colSize ) - zeroes;
}

void CSR::build_CSR( int**& matrix, int rowSize, int colSize ){

	allocate_CSR( matrix, rowSize, colSize );

	rowPtrs[ 0 ] = 0;

	int counter{ 0 };

	for( auto i = 0; i < rowSize; ++i ){

		for( auto j = 0; j < colSize; ++j ){

			// NOTE: any time we hit a 0, we use zeroes to put matrix[ i ][ j ] into the correct position in value, rowIdx, and colIdx
			if( matrix[ i ][ j ] != 0 ){

				value[ counter ] = matrix[ i ][ j ];
				colIdx[ counter ] = j;
				
				++counter;
			}
			
			// NOTE: If we hit a 0 in matrix, we increment the offset zeroes
			else{

				continue;
			}
		}

		rowPtrs[ i + 1 ] = counter;
	}
}

/// END - CSR

// ------------------

/// ELL - BEGIN

PairOfArrays ELL::nonzero_matrix( int**& matrix, int rowsize, int colsize ){

	PairOfArrays nonzeroes{ };
	nonzeroes.column = new int*[ rowsize ];
	nonzeroes.values = new int*[ rowsize ];
	nonzeroes.row_sizes = new int[ rowsize ]{ 0 };

	// Counts the amount of nonzeroes in each row of matrix. Stores the amount in nonzeroes_counter array
	for( auto i = 0; i < rowsize; ++i ){

		for( auto j = 0; j < colsize; ++j ){

			if( matrix[ i ][ j ] != 0 ){

				++nonzeroes.row_sizes[ i ];
			}

			else{

				continue;
			}
		}
	}

	for( auto i = 0; i < rowsize; ++i ){

		nonzeroes.column[ i ] = new int[ nonzeroes.row_sizes[ i ] ]{ 0 };
		nonzeroes.values[ i ] = new int[ nonzeroes.row_sizes[ i ] ]{ 0 };

		size += nonzeroes.row_sizes[ i ];
	}

	// NOTE: might have a problem when nonzeroes_counter[ i ] == 0.
	//		 Also - I could have put the below loops within the for loop at LOC 197, but
	//		 it would be hard to read.
	for( auto i = 0; i < rowsize; ++i ){

		int backward_counter = nonzeroes.row_sizes[ i ];

		for( auto j = 0; j < colsize; ++j ){

			if( matrix[ i ][ j ] != 0 ){

				nonzeroes.column[ i ][ nonzeroes.row_sizes[ i ] - backward_counter ] = j;
				nonzeroes.values[ i ][ nonzeroes.row_sizes[ i ] - backward_counter ] = matrix[ i ][ j ];
				
				--backward_counter;
			}

			else{

				continue;
			}
		}
	}

	return nonzeroes;
}

PairOfArrays ELL::padded_matrix( int**& matrix, int rowsize, int colsize ){

	PairOfArrays padded_mtx{ };
	padded_mtx.column = new int*[ rowsize ]{ nullptr };
	padded_mtx.values = new int*[ rowsize ]{ nullptr };

	PairOfArrays nonzeroes = nonzero_matrix( matrix, rowsize, colsize );

	int max_rowsize{ 0 };

	for( auto i = 0; i < rowsize; ++i ){

		if( nonzeroes.row_sizes[ i ] > max_rowsize ){

			max_rowsize = nonzeroes.row_sizes[ i ];
		}

		else{

			continue;
		}
	}

	
	for( auto i = 0; i < rowsize; ++i ){

		padded_mtx.column[ i ] = new int[ max_rowsize ]{ 0 };
		padded_mtx.values[ i ] = new int[ max_rowsize ]{ 0 };

		for( auto j = 0; j < nonzeroes.row_sizes[ i ]; ++j ){

			padded_mtx.column[ i ][ j ] = nonzeroes.column[ i ][ j ];
			padded_mtx.values[ i ][ j ] = nonzeroes.values[ i ][ j ];
		}
	}

	//std::copy( &padded_mtx[ 0 ][ 0 ], &padded_mtx[ 0 ][ 0 ] + ( max_rowsize * max_rowsize ), &nonzeroes.values[ 0 ][ 0 ] );
	//std::copy( &padded_columns[ 0 ][ 0 ], &padded_columns[ 0 ][ 0 ] + ( max_rowsize * max_rowsize ), &nonzeroes.column[ 0 ][ 0 ] );
	//padded_mtx.column = nonzeroes.column;
	//padded_mtx.values = nonzeroes.values;
	padded_mtx.row_sizes = new int{ max_rowsize };

	return padded_mtx;
}

// NOTE: verify if ELL::colIdx is correctly built!
void ELL::build_ELL( int**& matrix, int rowsize, int colsize ){

	PairOfArrays padded_mtx = padded_matrix( matrix, rowsize, colsize );

	colIdx = new int[ *padded_mtx.row_sizes * *padded_mtx.row_sizes ]{ 0 };
	value = new int[ *padded_mtx.row_sizes * *padded_mtx.row_sizes ]{ 0 };
	size = ( *padded_mtx.row_sizes * *padded_mtx.row_sizes );

	for( auto col = 0; col < *padded_mtx.row_sizes; ++col ){

		// NOTE: I think rowsize is wrong. E.g.: if matrix has one row full of zeroes, this row should be deleted when
		//		 nonzero_matrix is called, meaning that the nonzero matrix has rowsize - 1 rows.
		for( auto row = 0; row < rowsize; ++row ){
		
			// NOTE: this mapping WORKS but it is not right.
			colIdx[ ( col * *padded_mtx.row_sizes ) + row ] = padded_mtx.column[ row ][ col ];
			value[ ( col * *padded_mtx.row_sizes ) + row ] = padded_mtx.values[ row ][ col ];
		}
	}
}

/// END - ELL

// ------------------

/// JDS - BEGIN

PairOfArrays JDS::nonzero_matrix( int**& matrix, int rowsize, int colsize ){

	PairOfArrays nonzeroes{ };
	nonzeroes.column = new int*[ rowsize ]{ nullptr };
	nonzeroes.values = new int*[ rowsize ]{ nullptr };
	nonzeroes.row_sizes = new int[ rowsize ]{ 0 };

	// Counts the amount of nonzeroes in each row of matrix. Stores the amount in nonzeroes_counter array
	for( auto i = 0; i < rowsize; ++i ){

		for( auto j = 0; j < colsize; ++j ){

			if( matrix[ i ][ j ] != 0 ){

				++nonzeroes.row_sizes[ i ];
			}

			else{

				continue;
			}
		}
	}
	
	for( auto i = 0; i < rowsize; ++i ){

		nonzeroes.column[ i ] = new int[ nonzeroes.row_sizes[ i ] ]{ 0 };
		nonzeroes.values[ i ] = new int[ nonzeroes.row_sizes[ i ] ]{ 0 };
	}

	// NOTE: might have a problem when nonzeroes_counter[ i ] == 0.
	//		 Also - I could have put the below loops within the for loop at LOC 197, but
	//		 it would be hard to read.
	for( auto i = 0; i < rowsize; ++i ){

		int backward_counter = nonzeroes.row_sizes[ i ];

		for( auto j = 0; j < colsize; ++j ){

			if( matrix[ i ][ j ] != 0 ){

				nonzeroes.column[ i ][ nonzeroes.row_sizes[ i ] - backward_counter ] = j;
				nonzeroes.values[ i ][ nonzeroes.row_sizes[ i ] - backward_counter ] = matrix[ i ][ j ];
				
				--backward_counter;
			}

			else{

				continue;
			}
		}

		//backward_counter = 0;
	}

	return nonzeroes;
}

// NOTE: do NOT forget to sort JDS::row
void JDS::sort_rows( PairOfArrays& matrix, int rowsize ){

	for( auto i = 0; i < rowsize; ++i ){

		for( auto j = i + 1; j < rowsize; ++j ){

			if( matrix.row_sizes[ i ] < matrix.row_sizes[ j ] ){

				std::swap( matrix.column[ i ], matrix.column[ j ] );
				std::swap( matrix.values[ i ], matrix.values[ j ] );
				std::swap( row[ i ], row[ j ] ); // NOTE: maybe create a separate method for this?
				std::swap( matrix.row_sizes[ i ], matrix.row_sizes[ j ] ); // NOTE: this might be WRONG
			}

			else{

				continue;
			}
		}
	}
}

void JDS::build_matrix( int**& matrix, int rowsize, int colsize ){

	PairOfArrays nonzeroes = nonzero_matrix( matrix, rowsize, colsize );
	
	build_row( rowsize );

	sort_rows( nonzeroes, rowsize ); // NOTE: maybe let this method be private, since it is not intended to be called by outside code

	int max_rowsize = nonzeroes.row_sizes[ 0 ];
	int total_elements{ 0 };

	// NOTE: by sort_rows, rows are already sorted, meaning that max_rowsize = nonzeroes.row_size[ 0 ] already stores the largest row. This loop é redundant
	for( auto i = 0; i < rowsize; ++i ){

		// Meaning that I could delete this if-block
		if( max_rowsize < nonzeroes.row_sizes[ i ] ){

			max_rowsize = nonzeroes.row_sizes[ i ];
		}

		total_elements += nonzeroes.row_sizes[ i ];
	}

	size = total_elements;

	iterPtr = new int[ rowsize ]{ 0 };
	colIdx = new int[ total_elements ]{ 0 };
	value = new int[ total_elements ]{ 0 };

	int rowcounter{ 0 };
	int colcounter{ 0 };
	int ptr_counter{ 0 };

	int index{ 0 };

	while( colcounter < max_rowsize ){

		while( rowcounter < rowsize ){

			// Maybe the error is here?
			if( index < size ){

				colIdx[ index ] = nonzeroes.column[ rowcounter ][ colcounter ];
				value[ index ] = nonzeroes.values[ rowcounter ][ colcounter ];

				++index;
			}

			++rowcounter;
		}

		rowcounter = 0;

		++colcounter;
	}

	for( auto i = 1; i < rowsize; ++i ){

		ptr_counter += nonzeroes.row_sizes[ i ];

		iterPtr[ i ] = ptr_counter;
	}	
}

void JDS::build_row( int rowsize ){

	row = new int[ rowsize ]{ 0 };

	for( auto i = 0; i < rowsize; ++i ){

		row[ i ] = i;
	}
}

/// END - JDS