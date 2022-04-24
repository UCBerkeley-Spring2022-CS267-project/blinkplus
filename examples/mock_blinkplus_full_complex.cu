#include <stdio.h>
#include <array>
#include <string>
#include <cstdlib>
#include <vector>
#include <cstdlib>
#include <string>
#include "cuda_runtime.h"
#include "nccl.h"

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

struct group_info
{
    std::vector<int> devs;
    std::vector<ncclComm_t> comms;
    std::vector<int*> sendbuff;
    std::vector<int*> recvbuff;
    std::vector<cudaStream_t> streams;

    size_t num_comm;

    std::string graph_filepath;

    group_info( const char* graph_filepath_cstr, std::vector<int> devs )
    {
        if ( std::getenv(graph_filepath_cstr) == nullptr )
        {
            throw std::runtime_error("NCCL_GRAPH_FILE_CHAIN_021 not set\b");
        }
        graph_filepath = std::getenv(graph_filepath_cstr);

        this->devs = devs;
        this->num_comm = devs.size();
        this->resize( this->num_comm );
    }

    void resize( size_t num_comm )
    {
        comms.resize( num_comm );
        sendbuff.resize( num_comm );
        recvbuff.resize( num_comm );
        streams.resize( num_comm );
    }
};

void init_data( group_info& group, size_t data_size )
{
    for ( int i = 0; i < group.num_comm; ++i )
    {
        CUDACHECK(cudaSetDevice( group.devs[ i ] ));
        CUDACHECK(cudaMalloc( &(group.sendbuff[ i ]), data_size * sizeof(int)));
        CUDACHECK(cudaMalloc( &(group.recvbuff[ i ]), data_size * sizeof(int)));
        CUDACHECK(cudaMemset(  group.sendbuff[ i ], 1, data_size * sizeof(int)));
        CUDACHECK(cudaMemset(  group.recvbuff[ i ], 0, data_size * sizeof(int)));
        CUDACHECK(cudaStreamCreate( &(group.streams[ i ]) ));
    }
}

void init_data_helper( const group_info& group, group_info& helper_group, int helper_group_idx, int num_helper_group, size_t data_size )
{
    printf("\n");
    if ( helper_group_idx < 1 )
    {
        throw std::runtime_error("helper_group idx should start at 1\n");
    }

    for ( int i = 0; i < helper_group.num_comm; ++i )
    {
        printf("@LOG@ set helper group device %d\n", helper_group.devs[ i ] );
        CUDACHECK(cudaSetDevice( helper_group.devs[ i ] ));
    
        bool use_user_buffer = false;
        for ( int j = 0; j < group.num_comm; ++j )
        {
            // Use user group buffer 
            if ( group.devs[ j ] == helper_group.devs[ i ] )
            {
                printf("@LOG@ helper group device %d use user buffer\n", helper_group.devs[ i ] );
                use_user_buffer = true;

                int offset = helper_group_idx * data_size / (num_helper_group+1);
                helper_group.sendbuff[ i ] = group.sendbuff[ i ] + offset;
                helper_group.recvbuff[ i ] = group.recvbuff[ i ] + offset;

                //CUDACHECK(cudaMalloc( &(helper_group.sendbuff[ i ]), data_size / (num_helper_group+1) * sizeof(int)));
                //CUDACHECK(cudaMalloc( &(helper_group.recvbuff[ i ]), data_size / (num_helper_group+1) * sizeof(int)));

                //CUDACHECK(cudaMalloc( &(helper_group.recvbuff[ i ]), data_size / (num_helper_group+1) * sizeof(int)));
                //CUDACHECK(cudaMemset(  helper_group.recvbuff[ i ], 0, data_size / (num_helper_group+1) * sizeof(int)));

                printf("@LOG@ user address %lld/%p, our address %lld/%p, address diff %lld, offset %d, data size %d, helper idx %d\n", \
                    group.sendbuff[ i ], group.sendbuff[ i ], helper_group.sendbuff[ i ], helper_group.sendbuff[ i ], \
                    helper_group.sendbuff[ i ] - group.sendbuff[ i ], \
                    offset, data_size, helper_group_idx );
                break;
            }
        }

        // allocate own buffer
        if ( !use_user_buffer )
        {
            printf("@LOG@ helper group device %d use internal buffer\n", helper_group.devs[ i ] );
            CUDACHECK(cudaMalloc( &(helper_group.sendbuff[ i ]), data_size / (num_helper_group+1) * sizeof(int)));
            CUDACHECK(cudaMalloc( &(helper_group.recvbuff[ i ]), data_size / (num_helper_group+1) * sizeof(int)));
            CUDACHECK(cudaMemset(  helper_group.sendbuff[ i ], 1, data_size / (num_helper_group+1) * sizeof(int)));
            CUDACHECK(cudaMemset(  helper_group.recvbuff[ i ], 0, data_size / (num_helper_group+1) * sizeof(int)));
        }

        CUDACHECK(cudaStreamCreate( &(helper_group.streams[ i ]) ));
    }
}

void run_broadcast( group_info& group, size_t data_size )
{
    NCCLCHECK(ncclGroupStart());
    for ( int i = 0; i < group.num_comm; ++i ) 
    {
        NCCLCHECK(ncclBroadcast((const void*)group.sendbuff[ i ], \
                                (void*)group.recvbuff[ i ], \
                                data_size, ncclInt, 0, \
                                group.comms[i], \
                                group.streams[i]));
    }
    NCCLCHECK(ncclGroupEnd());
}

void run_reduce( group_info& group, size_t data_size )
{
    NCCLCHECK(ncclGroupStart());
    for ( int i = 0; i < group.num_comm; ++i ) 
    {
      // allreduce
      NCCLCHECK(ncclAllReduce( (const void*)group.sendbuff[ i ], \
                               (void*)group.recvbuff[ i ], \
                               data_size, ncclInt, ncclSum, \
                               group.comms[i], \
                               group.streams[i]) );
    }
    NCCLCHECK(ncclGroupEnd());
}

void init_comm( group_info& group )
{
    setenv( "NCCL_GRAPH_FILE", group.graph_filepath.c_str() , 1 );
    NCCLCHECK(ncclCommInitAll( group.comms.data(), group.num_comm, group.devs.data() ));
}

void sync_stream( group_info& group )
{
    for ( int i = 0; i < group.num_comm; ++i ) 
    {
      CUDACHECK(cudaSetDevice( group.devs[i] ));
      CUDACHECK(cudaStreamSynchronize( group.streams[i] ));
    }
}

void free_buffer( group_info& group )
{
    for ( int i = 0; i < group.num_comm; ++i ) 
    {
        CUDACHECK(cudaSetDevice( group.devs[i] ));
        CUDACHECK(cudaFree( group.sendbuff[i] ));
        CUDACHECK(cudaFree( group.recvbuff[i] ));
    }
}

void free_buffer_helper( const group_info& group, group_info& helper_group )
{
    for ( int i = 0; i < helper_group.num_comm; ++i ) 
    {
        bool is_user_buffer = false;
        for ( int j = 0; j < group.num_comm; ++j )
        {
            if ( group.devs[ j ] == helper_group.devs[ i ] )
            {
                is_user_buffer = true;
                break;
            }
        }
        
        if ( !is_user_buffer )
        {
            CUDACHECK(cudaSetDevice( helper_group.devs[i] ));
            CUDACHECK(cudaFree( helper_group.sendbuff[i] ));
            CUDACHECK(cudaFree( helper_group.recvbuff[i] ));
        }
    }
}


void free_nccl( group_info& group )
{
    for ( int i = 0; i < group.num_comm; ++i ) 
    {
        ncclCommDestroy( group.comms[i]);
    }
}

int main(int argc, char* argv[])
{
    // Reference
    // https://github.com/NVIDIA/nccl/issues/574

    // set enviroment variable before run
    // this is program level setting and thus do not pollute global 
    setenv( "NCCL_PROTO", "Simple", 1);
    //setenv( "NCCL_DEBUG", "Info", 1);
    //setenv( "NCCL_DEBUG_SUBSYS", "ALL", 1);
    setenv( "NCCL_ALGO", "Tree", 1 ); // Tree : AllReduceTree+BroadcastRing | Ring : AllReduceRing+BroadcastRing

    // managing 4 devices
    int data_size = 256*1024*1024;
    int num_helper_group = 1;

    group_info group01( "NCCL_GRAPH_FILE_CHAIN_01", std::vector<int>{0,1} );
    group_info helper_group021( "NCCL_GRAPH_FILE_CHAIN_021", std::vector<int>{0,2,1} );
    // group_info helper_group031( "NCCL_GRAPH_FILE_CHAIN_031", std::vector<int>{0,3,1} );
    // group_info helper_group0321( "NCCL_GRAPH_FILE_CHAIN_0321", std::vector<int>{0,3,2,1} );

    // Set and initial data
    init_data( group01, data_size );
    init_data_helper( group01, helper_group021, 1, num_helper_group, data_size );
    // init_data_helper( group01, helper_group031, 2, num_helper_group, data_size );
    // init_data_helper( group01, helper_group0321, 3, num_helper_group, data_size );

    // Initial communicator
    printf("\n\n!!!!!Initial comm\n"); fflush(stdout);
    init_comm( group01 );
    init_comm( helper_group021 );
    // init_comm( helper_group031 );
    // init_comm( helper_group0321 );

    // Collective run
    printf("\n\n!!!!!Run broadcast\n"); fflush(stdout);
    run_broadcast( group01, data_size / ( num_helper_group + 1 ) );
    run_broadcast( helper_group021, data_size / ( num_helper_group + 1 ));
    // run_broadcast( helper_group031, data_size / ( num_helper_group + 1 ) );
    // run_broadcast( helper_group0321, data_size / ( num_helper_group + 1 ) );

    // printf("\n\n!!!!!Run allreduce\n"); fflush( stdout );
    // run_reduce( group01, data_size / ( num_helper_group + 1 ) );
    // run_reduce( helper_group021, data_size / ( num_helper_group + 1 ) );
    // run_reduce( helper_group031, data_size / ( num_helper_group + 1 ) );
    // run_reduce( helper_group0321, data_size / ( num_helper_group + 1 ) );

    // synchronize streams
    printf("\n\n!!!!!stream synchronize user\n"); fflush(stdout);
    sync_stream( group01 );
    printf("\n\n!!!!!stream synchronize helper 1\n"); fflush(stdout);
    sync_stream( helper_group021 );
    printf("\n\n!!!!!stream synchronize helper 2\n"); fflush(stdout);
    // sync_stream( helper_group031 );
    printf("\n\n!!!!!stream synchronize helper 3\n"); fflush(stdout);
    // sync_stream( helper_group0321 );

    //free device buffers
    printf("\n\n!!!!!free used buffer\n"); fflush(stdout);
    free_buffer( group01 );
    free_buffer_helper( group01, helper_group021 );
    // free_buffer_helper( group01, helper_group031 );
    // free_buffer_helper( group01, helper_group0321 );

    //finalizing NCCL
    printf("\n\n!!!!!free comm buffer\n"); fflush(stdout);
    free_nccl( group01 );
    free_nccl( helper_group021 );
    // free_nccl( helper_group031 );
    // free_nccl( helper_group0321 );

    printf("\n\n!!!!!Success \n");
    return 0;
}