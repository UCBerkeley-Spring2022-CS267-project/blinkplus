#include <stdio.h>
#include <array>
#include <string>
#include <cstdlib>
#include <vector>
#include <cstdlib>
#include <string>
#include "cuda_runtime.h"
#include "nccl.h"
//#include "cuda_profiler_api.h"

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
        CUDACHECK(cudaStreamCreateWithFlags( &(group.streams[i]), cudaStreamNonBlocking ));
    }
}

__global__ void dummy_kernel()
{
    printf("dummy run\n");
    return;
}

void run_dummy( group_info& group )
{
    for ( int i = 0; i < group.num_comm; ++i ) 
    {
        CUDACHECK(cudaSetDevice( group.devs[i] ));
        // __host__ ​cudaError_t cudaLaunchKernel ( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream )
        CUDACHECK( cudaLaunchKernel( (void*)&dummy_kernel, dim3(1), dim3(1), nullptr, 0, group.streams[ i ] ) );
    }
}

void run_broadcast( group_info& group, size_t data_size, bool forward = true )
{
    printf("\n@LOG@ run_broadcast start\n"); fflush(stdout);
    NCCLCHECK(ncclGroupStart());
    printf("@LOG@ ncclBroadcast start\n"); fflush(stdout);
    for ( int i = 0; i < group.num_comm; ++i ) 
    {
        printf("@LOG@ ncclBroadcast %d/%d\n", i, group.num_comm); fflush(stdout);
        NCCLCHECK(ncclBroadcast((const void*)group.sendbuff[ i ], \
                                (void*)group.recvbuff[ i ], \
                                data_size, ncclInt, 0, \
                                group.comms[i], \
                                group.streams[i]));
    }
    printf("@LOG@ ncclBroadcast end\n"); fflush(stdout);
    NCCLCHECK(ncclGroupEnd());
    printf("@LOG@ run_broadcast end\n"); fflush(stdout);
}

void run_reduce( group_info& group, size_t data_size )
{
    printf("\n@LOG@ run_reduce start\n"); fflush(stdout);
    NCCLCHECK(ncclGroupStart());
    printf("@LOG@ ncclAllReduce start\n"); fflush(stdout);
    for ( int i = 0; i < group.num_comm; ++i ) 
    {
        printf("@LOG@ ncclAllReduce %d/%d\n", i, group.num_comm); fflush(stdout);
      // allreduce
      NCCLCHECK(ncclAllReduce( (const void*)group.sendbuff[ i ], \
                               (void*)group.recvbuff[ i ], \
                               data_size, ncclInt, ncclSum, \
                               group.comms[i], \
                               group.streams[i]) );
    }
    printf("@LOG@ ncclAllReduce end\n"); fflush(stdout);
    NCCLCHECK(ncclGroupEnd());
    printf("@LOG@ run_reduce end\n"); fflush(stdout);
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
      CUDACHECK(cudaSetDevice( group.devs[i]));
      CUDACHECK(cudaStreamSynchronize( group.streams[i]));
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

void free_nccl( group_info& group )
{
    for ( int i = 0; i < group.num_comm; ++i ) 
    {
        ncclCommDestroy( group.comms[i]);
    }
}

int main(int argc, char* argv[])
{
    printf("NCCL Version %d.%d.%d\n", NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH );

    // Reference
    // https://github.com/NVIDIA/nccl/issues/574
    // https://github.com/NVIDIA/nccl/issues/217
    // https://github.com/NVIDIA/nccl/issues/195#issuecomment-473344810
    // https://github.com/NVIDIA/nccl/issues/239#issuecomment-510565429
    // https://github.com/NVIDIA/nccl/issues/315

    // set enviroment variable before run
    // this is program level setting and thus do not pollute global 
    setenv( "NCCL_PROTO", "Simple", 1);
    setenv( "NCCL_DEBUG", "Info", 1);
    setenv( "NCCL_DEBUG_SUBSYS", "ALL", 1);
    setenv( "NCCL_ALGO", "Ring", 1 ); // Tree : AllReduceTree+BroadcastRing | Ring : AllReduceRing+BroadcastRing

    // managing 4 devices
    int data_size = 16*1024*1024;

    group_info group01( "NCCL_GRAPH_FILE_CHAIN_01", std::vector<int>{0,1} );
    group_info group02( "NCCL_GRAPH_FILE_CHAIN_02", std::vector<int>{0,2} );
    group_info group03( "NCCL_GRAPH_FILE_CHAIN_03", std::vector<int>{0,3} );

    // Set and initial data
    init_data( group01, data_size );
    init_data( group02, data_size );
    init_data( group03, data_size );

    // Initial communicator
    printf("@LOG@ Initial comm\n"); fflush(stdout);
    init_comm( group01 );
    init_comm( group02 );
    init_comm( group03 );

    // Collective run
    printf("@LOG@ Run collective 1\n"); fflush(stdout);
    run_broadcast( group01, data_size/10 );
    run_broadcast( group01, data_size/10 );

    printf("@LOG@ Run collective 2\n"); fflush(stdout);
    run_reduce( group01, data_size/10 );

    printf("@LOG@ Run collective 3\n"); fflush(stdout);
    run_broadcast( group02, data_size/10 );

    printf("@LOG@ Run collective 4\n"); fflush(stdout);
    run_reduce( group02, data_size/10 );

    printf("@LOG@ Run collective 5\n"); fflush(stdout);
    run_broadcast( group03, data_size/10 );

    printf("@LOG@ Run collective 6\n"); fflush(stdout);
    run_reduce( group03, data_size/10 );

    // synchronize streams
    printf("@LOG@ stream synchronize 1\n"); fflush(stdout);
    sync_stream( group01 );

    printf("@LOG@ stream synchronize 2\n"); fflush(stdout);
    sync_stream( group02 );

    printf("@LOG@ stream synchronize 2\n"); fflush(stdout);
    sync_stream( group03 );

    //free device buffers
    printf("@LOG@ free used buffer\n"); fflush(stdout);
    free_buffer( group01 );
    free_buffer( group02 );
    free_buffer( group03 );

    //finalizing NCCL
    printf("@LOG@ free comm buffer\n"); fflush(stdout);
    free_nccl( group01 );
    free_nccl( group02 );
    free_nccl( group03 );

    printf("@LOG@ Success \n");
    return 0;
}