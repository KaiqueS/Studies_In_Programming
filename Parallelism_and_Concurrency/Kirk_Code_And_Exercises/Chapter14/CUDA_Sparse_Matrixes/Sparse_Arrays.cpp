#include "Sparse_Arrays.hpp"

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

	rowIdx = new int[ ( rowSize * colSize ) - zeroes ];
	colIdx = new int[ ( rowSize * colSize ) - zeroes ];
	value = new int[ ( rowSize * colSize ) - zeroes ];
}

void COO::build_COO( int**& matrix, int rowSize, int colSize ){

	zeroes = count_Zeroes( matrix, rowSize, colSize );

	allocate_COO( matrix, rowSize, colSize );

	for( auto i = 0; i < rowSize; ++i ){

		for( auto j = 0; j < colSize; ++j ){

			// NOTE: any time we hit a 0, we use zeroes to put matrix[ i ][ j ] into the correct position in value, rowIdx, and colIdx
			if( ( matrix[ i ][ j ] != 0 ) && ( ( ( i * rowSize ) + j ) < ( ( rowSize * colSize ) - zeroes ) ) ){

				rowIdx[ ( i * rowSize ) + j ] = i;
				colIdx[ ( i * rowSize ) + j ] = j;
				value[ ( i * rowSize ) + j ] = matrix[ i ][ j ];
			}
			
			else{

				continue;
			}
		}
	}

	size = ( rowSize * colSize ) - zeroes;
}

void COO::insert_element( int row, int col, int element ){

	int* newRow = new int[ size + 1 ];
	int* newCol = new int[ size + 1 ];
	int* newVal = new int[ size + 1 ];

	std::copy( rowIdx[ 0 ], rowIdx[ size - 1 ], newRow[ 0 ] );
	std::copy( colIdx[ 0 ], colIdx[ size - 1 ], newCol[ 0 ] );
	std::copy( value[ 0 ], value[ size - 1 ], newVal[ 0 ] );

	newRow[ size ] = row;
	newCol[ size ] = col;
	newVal[ size ] = element;

	++size;

	rowIdx = std::move( newRow );
	colIdx = std::move( newCol );
	value = std::move( newVal );

	delete[ ] newRow, newCol, newVal;
}

void COO::reorder( int left_index, int right_index ){

	std::swap( rowIdx[ left_index ], rowIdx[ right_index ] );
	std::swap( colIdx[ left_index ], colIdx[ right_index ] );
	std::swap( value[ left_index ], value[ right_index ] );
}

/// END - COO

// ------------------

/// CSR - BEGIN

int CSR::count_Zeroes( int**& matrix, int rowSize, int colSize ){

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

	rowPtrs = new int[ rowSize + 1 ];
	colIdx = new int[ ( rowSize * colSize ) - zeroes ];
	value = new int[ ( rowSize * colSize ) - zeroes ];

	size = ( rowSize * colSize ) - zeroes;
}

void CSR::build_CSR( int**& matrix, int rowSize, int colSize ){

	allocate_CSR( matrix, rowSize, colSize );

	rowPtrs[ 0 ] = 0;

	for( auto i = 0; i < rowSize; ++i ){

		for( auto j = 0; j < colSize; ++j ){

			// NOTE: any time we hit a 0, we use zeroes to put matrix[ i ][ j ] into the correct position in value, rowIdx, and colIdx
			if( ( matrix[ i ][ j ] != 0 ) && ( ( ( i * rowSize ) + j ) < ( ( rowSize * colSize ) - zeroes ) ) ){

				value[ ( i * rowSize ) + j ] = ( matrix[ i ][ j ] );
				colIdx[ ( i * rowSize ) + j ] = j;

				// NOTE: this means that, if rowPtrs[ x ] == rowPtrs[ x + 1 ], then row x has no nonzero elements
				rowPtrs[ i + 1 ] += 1;
			}
			
			// NOTE: If we hit a 0 in matrix, we increment the offset zeroes
			else{

				continue;
			}
		}
	}
}

/// END - CSR

// ------------------

/// ELL - BEGIN

PairOfArrays ELL::nonzero_matrix( int**& matrix, int rowsize, int colsize ){

	PairOfArrays nonzeroes{ };

	//int** nonzeroes{ };

	nonzeroes.row_sizes = new int[ rowsize ];

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

		nonzeroes.column = new int*[ nonzeroes.row_sizes[ i ] ];
		nonzeroes.values = new int*[ nonzeroes.row_sizes[ i ] ];
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

		backward_counter = 0;
	}

	return nonzeroes;
}

PairOfArrays ELL::padded_matrix( int**& matrix, int rowsize, int colsize ){

	int** padded_mtx{ };
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

	for( auto i = 0; i < nonzeroes.size( ); ++i ){

		if( nonzeroes[ i ].size( ) < max_rowsize ){

			std::vector<int> holder( max_rowsize, 0 );
			std::copy( std::begin( nonzeroes[ i ] ), std::end( nonzeroes[ i ] ), std::begin( holder ) );
			//holder.resize( max_rowsize );
			//std::fill( ( holder.begin() + nonzeroes[ i ].size( ) ), holder.end( ), 0 );
			padded_mtx.push_back( holder );
		}

		else{

			padded_mtx.push_back( nonzeroes[ i ] );
		}
	}

	return padded_mtx;
}

// NOTE: verify if ELL::colIdx is correctly built!
void ELL::build_ELL( int**& matrix, int rowsize, int colsize ){

	std::vector<std::vector<int>> padded_mtx = padded_matrix( matrix, rowsize, colsize );

	for( auto col = 0; col < colsize; ++col ){

		for( auto row = 0; row < rowsize; ++row ){
		
			colIdx.push_back( col );
			value.push_back( padded_mtx[ row ][ col ] );
		}
	}
}

/// END - ELL

// ------------------

/// JDS - BEGIN

std::vector<std::vector<int>> JDS::nonzero_matrix( int**& matrix, int rowsize, int colsize ){

	std::vector<std::vector<int>> nonzeroes{ };

	for( auto i = 0; i < rowsize; ++i ){

		std::vector<int> row{ };

		for( auto j = 0; j < colsize; ++j ){

			if( matrix[ i ][ j ] != 0 ){

				row.push_back( matrix[ i ][ j ] );
			}

			else{

				continue;
			}
		}

		nonzeroes.push_back( row );
	}

	return nonzeroes;
}

std::vector<std::vector<int>> JDS::nonzero_colidx( int**& matrix, int rowsize, int colsize ){

	std::vector<std::vector<int>> nonzeroes{ };

	for( auto i = 0; i < rowsize; ++i ){

		std::vector<int> row{ };

		for( auto j = 0; j < colsize; ++j ){

			if( matrix[ i ][ j ] != 0 ){

				row.push_back( j );
			}

			else{

				continue;
			}
		}

		nonzeroes.push_back( row );
	}

	return nonzeroes;
}

// NOTE: do NOT forget to sort JDS::row
void JDS::sort_rows( std::vector<std::vector<int>>& matrix ){

	for( auto i = 0; i < matrix.size( ); ++i ){

		for( auto j = i + 1; j < matrix.size( ); ++j ){

			if( matrix[ i ].size( ) < matrix[ j ].size( ) ){

				std::swap( matrix[ i ], matrix[ j ] );
				std::swap( row[ i ], row[ j ] ); // NOTE: maybe create a separate method for this?
			}

			else{

				continue;
			}
		}
	}
}

void JDS::build_matrix( int**& matrix, int rowsize, int colsize ){

	std::vector<std::vector<int>> nonzeroes = nonzero_matrix( matrix, rowsize, colsize );
	std::vector<std::vector<int>> col_nonzeroes = nonzero_colidx( matrix, rowsize, colsize );
	
	build_row( nonzeroes );

	sort_rows( nonzeroes ); // NOTE: maybe let this method be private, since it is not intended to be called by outside code
	sort_rows( col_nonzeroes );

	int rowcounter{ 0 };
	int colcounter{ 0 };

	// NOTE: since nonzeroes is sorted in descending order, for any 0 <= i < nonzeroes.size( ), we have that the vector
	//		 nonzeroes[ i ].size( ) >= nonzeroes[ i + 1 ].size( )
	// NOTE: maybe check for nullptr? In case nonzeroes.empty == true.
	while( colcounter < nonzeroes.begin( ) -> size( ) ){

		for( std::vector<int> row : nonzeroes ){

			if( colcounter < row.size( ) ){

				value.push_back( row[ colcounter ] );
			}

			else{

				continue;
			}
		}

		for( std::vector<int> row : col_nonzeroes ){

			if( colcounter < row.size( ) ){

				colIdx.push_back( row[ colcounter ] );
			}

			else{

				continue;
			}
		}

		++colcounter;
	}
}

void JDS::build_row( std::vector<std::vector<int>>& matrix ){

	for( auto i = 0; i < matrix.size( ); ++i ){

		row.push_back( i );
	}
}

/// END - JDS

// ------------------