#include <mpi.h>
#include <vector>
#include <iostream>
#include <random>

// 3.11 Finding prefix sums is a generalization of global sum. Rather than simply finding the sum of n values,
// x0 + x1 + · · · + xn−1, the prefix sums are the n partial sums x0 , x0 + x1 , x0 + x1 + x2 ,. . . , x0 + x1 + · · · + xn−1 .

// a. Devise a serial algorithm for computing the n prefix sums of an array with n elements.

std::vector<int> serial_prefix_sum( std::vector<int>& elements ){

    int partial{ 0 };
    std::vector<int> partial_sums{ };

    for( std::vector<int>::size_type i = 0; i < elements.size( ); ++i ){

        partial += elements[ i ];
        partial_sums.push_back( partial );
    }
     
     return partial_sums;
}

// b. Parallelize your serial algorithm for a system with n processes, each of which is storing one of the x_i’s.

std::vector<int> build_vector( int my_rank, int size ){

    std::vector<int> elements( size );

    if( my_rank == 0 ){

        std::random_device rand;
        std::uniform_int_distribution dist( 0, size );
        std::mt19937_64 rng( rand( ) );

        for( std::vector<int>::size_type i = 0; i < elements.size( ); ++i ){

            elements[ i ] = dist( rng );
        }
    }

    return elements;
}

void share_data( int my_rank, std::vector<int> send ,std::vector<int>& receive, MPI_Comm comm ){

    //MPI_Bcast( send.data( ), send.size( ), MPI_INT, 0, MPI_COMM_WORLD );
    if( my_rank == 0 ){

        MPI_Scatter( send.data( ), receive.size( ), MPI_INT, receive.data( ), receive.size( ), MPI_INT, my_rank, comm );

        send.clear( );
    }

    else{

        MPI_Scatter( send.data( ), receive.size( ), MPI_INT, receive.data( ), receive.size( ), MPI_INT, 0, comm );
    }
}

// I will perform a broadcast of the whole vector. But each process will only perform their respective partial sums
void parallel_prefix_sum( int my_rank, std::vector<int>& elements ){

    
}

void add_scalar( int scalar, std::vector<int>& elements ){

    for( std::vector<int>::size_type i = 0; i < elements.size( ); ++i ){

        elements[ i ] += scalar;
    }
 }

int previous_node( int connection, int comm_size, int iteration ){

    if( connection == 0 ){

        return connection;
    }

    else if( iteration == 0 ){

        return connection - 1;
    }

    else{

        connection = previous_node( connection - 1, comm_size, iteration - 1 );

        return connection;
    }
}

int next_node( int connection, int comm_size, int iteration ){

    if( connection == comm_size - 1 ){

        return connection;
    }

    else if( iteration == 0 ){

        return connection + 1;
    }

    else{

        connection = next_node( connection + 1, comm_size, iteration - 1 );

        //return next_node( connection, comm_size, iteration - 1 );

        return connection;
    }
}


class Comm_Node{

public:

    Comm_Node( ){ }
    Comm_Node( int val, Comm_Node* pred, Comm_Node* succ ) : my_rank( val ), predecessor( pred ), successor( succ ){ }
    ~Comm_Node( ){ }

    void set_rank( int val ){ my_rank = val; }
    void set_pred( Comm_Node* pred ){ predecessor = pred; };
    void set_succ( Comm_Node* succ ){ successor = succ; }

    int rank( ){ return my_rank; }
    Comm_Node* pred( ){ return predecessor; }
    Comm_Node* succ( ){ return successor; }

private:

    int my_rank{ 0 };
    Comm_Node* predecessor{ nullptr };
    Comm_Node* successor{ nullptr };
};

class Comm_Graph{

public:

    Comm_Graph( ){ }
    Comm_Graph( int comm_sz ){

        nodes.resize( comm_sz );

        for( std::vector<int>::size_type i = 0; i < comm_sz; ++i ){

            nodes[ i ].set_rank( i );
            nodes[ i ].set_pred( nullptr );
            nodes[ i ].set_succ( nullptr );
        }
    }
    ~Comm_Graph( ){ nodes.clear( ); }

    void build_graph( ){
        
        for( std::vector<Comm_Node*>::size_type i = 0; i < nodes.size( ); ++i ){

            if( i == 0 ){

                nodes[ i ].set_succ( &nodes[ i + 1 ] );
            }

            else if( i == nodes.size( ) - 1 ){

                nodes[ i ].set_pred( &nodes[ i - 1 ] );
            }

            else{
                
                nodes[ i ].set_pred( &nodes[ i - 1 ] );
                nodes[ i ].set_succ( &nodes[ i + 1 ] );
            }
        }
    }

    // PROBLEM HERE -> SOMEHOW I NEED TO PRESERVE ITERATORS IN BETWEEN ITERATIONS IN THE FOR LOOP
    void update( ){

        for( auto i = 0; i < this -> size( ); ++i ){

            if( nodes[ i ].pred( ) != nullptr && nodes[ i ].succ( ) != nullptr ){

                nodes[ i ].set_pred( nodes[ i ].pred( ) -> pred( ) );
                nodes[ i ].set_succ( nodes[ i ].succ( ) -> succ( ) );
            }

            else if( nodes[ i ].pred( ) != nullptr ){

                nodes[ i ].set_pred( nodes[ i ].pred( ) -> pred( ) );
            }

            else if( nodes[ i ].succ( ) != nullptr ){

                nodes[ i ].set_succ( nodes[ i ].succ( ) -> succ( ) );
            }

            else{

                continue;
            }
        }
    }

    Comm_Node& operator[ ]( std::vector<int>::size_type index ){

        return nodes[ index ];
    }

    int size( ){ return nodes.size( ); }

private:

    std::vector<Comm_Node> nodes{ };
};

// Represent the processes as nodes in a graph that is a tree. At the start, on the tree, each node has an edge
// from itself to the next node, except for the last node, which is linked to no other node. Assign to each process
// ( vector size / comm_sz ) elements, and let each node perform a serial prefix sum on their own elements. After
// this sum, send the GREATEST element from each node to the node its linked to. Sum this new element to its own
// elements. Now, for nodes i - 1, i, and i + 1, remove the edges ( i - 1, i ) and ( i, i + 1 ), and add the edge
// ( i - 1, i + 1 ), i.e., let node i - 1 be linked to its successor's successor. Repeat the sending of the greatest
// element. Repeat edge deletion and insertion until the LAST node becomes the successor of the FIRST node. Do one
// more sum. QED
std::vector<int> parallel_prefix_sum_graph( int my_rank, int comm_sz, std::vector<int>& elements, MPI_Comm comm ){

    Comm_Graph graph( comm_sz );
    graph.build_graph( );

    std::vector<int> local_partial = serial_prefix_sum( elements );

    for( auto i = 0; i < graph.size( ); ++i ){

        if( graph[ i ].rank( ) == my_rank ){

            while( graph[ i ].pred( ) != nullptr || graph[ i ].succ( ) != nullptr ){

                int val = local_partial[ local_partial.size( ) - 1 ];

                if( graph[ i ].pred( ) != nullptr && graph[ i ].succ( ) != nullptr ){

                    MPI_Send( &val, 1, MPI_INT, graph[ i ].succ( ) -> rank( ), 0, comm );
                    MPI_Recv( &val, 1, MPI_INT, graph[ i ].pred( ) -> rank( ), 0, comm, MPI_STATUS_IGNORE );

                    add_scalar( val, local_partial );
                }

                else if( graph[ i ].pred( ) != nullptr ){

                    MPI_Recv( &val, 1, MPI_INT, graph[ i ].pred( ) -> rank( ), 0, comm, MPI_STATUS_IGNORE );

                    add_scalar( val, local_partial );
                }

                else if( graph[ i ].succ( ) != nullptr ){

                    MPI_Send( &val, 1, MPI_INT, graph[ i ].succ( ) -> rank( ), 0, comm );
                }

                else{

                    continue;
                }

                graph.update( );
            }
        }

        else{
            
            continue;
        }
    }

    std::vector<int> result( elements.size( ) * comm_sz );

    MPI_Gather( local_partial.data( ), local_partial.size( ), MPI_INT, result.data( ), local_partial.size( ), MPI_INT, 0, comm );

    return result;
}

std::vector<int> parallel_prefix_sum_enhanced( int my_rank, int comm_sz, std::vector<int>& elements, MPI_Comm comm ){

    //int local_n = elements.size( ) / comm_sz;

    std::vector<int> local_partial{ };

    // MPI_Scatter( elements.data( ), local_n, MPI_INT, local_partial.data( ), local_n, MPI_INT, 0, comm );

    local_partial = serial_prefix_sum( elements );

    for( std::vector<int>::size_type i = 1; i < comm_sz; ++i ){

        int value{ 0 };

        value = local_partial[ local_partial.size( ) - 1 ];

        if( my_rank == comm_sz - 1 ){

            MPI_Recv( &value, 1, MPI_INT, previous_node( my_rank, comm_sz, i ), 0, comm, MPI_STATUS_IGNORE );
            add_scalar( value, local_partial );
        }

        else if( my_rank == 0 ){

            MPI_Send( &value, 1, MPI_INT, next_node( my_rank, comm_sz, i ), 0, comm );
        }

        else if( my_rank - i >= 0 ){

            MPI_Send( &value, 1, MPI_INT, next_node( my_rank, comm_sz, i ), 0, comm );
            MPI_Recv( &value, 1, MPI_INT, previous_node( my_rank, comm_sz, i ), 0, comm, MPI_STATUS_IGNORE );

            add_scalar( value, local_partial );

            value = local_partial[ local_partial.size( ) - 1 ];
        }        
    }

    std::vector<int> result( elements.size( ) * comm_sz );

    MPI_Gather( local_partial.data( ), local_partial.size( ), MPI_INT, result.data( ), local_partial.size( ), MPI_INT, 0, comm );

    return result;
}

// c. Suppose n = 2k for some positive integer k. Can you devise a serial algorithm and a parallelization of the serial algorithm
// so that the parallel algorithm requires only k communication phases? (You might want to look for this online.)

// d. MPI provides a collective communication function, MPI_Scan, that can be used to compute prefix sums:

// int MPI_Scan ( void∗ sendbuf_p, void∗ recvbuf_p, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm ) ;

// It operates on arrays with count elements; both sendbuf_p and recvbuf_p should refer to blocks of count elements of type datatype.
// The op argument is the same as op for MPI_Reduce. Write an MPI program that generates a random array of count elements on each MPI
// process, finds the prefix sums, and prints the results.

int main( ){

    int my_rank{ 0 }, comm_sz{ 0 };

    MPI_Init( nullptr, nullptr );
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &comm_sz );

    //std::vector<int> elem{ 1, 2, 3, 4, 5 };
    //std::vector<int> sums = serial_prefix_sum( elem );

    int vec_size{ 0 };

    if( my_rank == 0 ){ 

        printf( "Enter the size of the vector: " );
        scanf( "%d", &vec_size );
    }

    MPI_Bcast( &vec_size, 1, MPI_INT, 0, MPI_COMM_WORLD );

    std::vector<int> send_elements = build_vector( my_rank, vec_size );
    std::vector<int> receive_elements( send_elements.size( ) / comm_sz );

    share_data( my_rank, send_elements, receive_elements, MPI_COMM_WORLD );

    std::vector<int> result = parallel_prefix_sum_graph( my_rank, comm_sz, receive_elements, MPI_COMM_WORLD );

    if( my_rank == 0 ){

        for( auto i = 0; i < send_elements.size( ); ++i ){

            std::cout << send_elements[ i ] << " ";
        }

        std::cout << "\n";

        for( auto i = 0; i < result.size( ); ++i ){

            std::cout << result[ i ] << " ";
        }

        std::cout << "\n";
    }

    MPI_Finalize( );
}