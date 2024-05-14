#include <iostream>
#include <vector>
#include <mpi.h>
#include <algorithm>

void Merge_low( std::vector<int> my_keys, std::vector<int> recv_keys, std::vector<int> temp_keys, int local_n ){

    int m_i{ 0 }, r_i{ 0 }, t_i{ 0 };

    while( t_i < local_n ){

        if( my_keys[ m_i ] <= recv_keys[ r_i ] ){

            temp_keys[ t_i ] = my_keys[ m_i ];

            ++t_i;
            ++m_i;
        }

        else{

            temp_keys[ t_i ] = recv_keys[ r_i ];
            ++t_i;
            ++r_i;
        }
    }

    for( m_i = 0; m_i < local_n; ++m_i ){

        my_keys[ m_i ] = temp_keys[ m_i ];
    }
}

int compare( const void* left, const void* right ){

    if( left < right ){

        return left;
    }

    else{

        return right;
    }
}

void odd_even_sort( std::vector<int> local_keys, MPI_Comm comm ){

    std::sort( local_keys.begin( ), local_keys.end( ) );
}

int main( ){


}