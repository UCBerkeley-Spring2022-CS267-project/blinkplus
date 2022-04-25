#include <stdio.h>
#include <array>
#include <string>
#include <cstdlib>
#include <vector>
#include <cstdlib>
#include <string>
#include "cuda_runtime.h"
#include "nccl.h"
#include "blinkplus.h"

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


int main(int argc, char* argv[]) // char** argv
{
    printf("NCCL Version %d.%d.%d\n", NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH );

    // NCCL enviroment varaible
    // setenv( "NCCL_PROTO", "Simple", 1);
    // setenv( "NCCL_DEBUG", "Info", 1);
    // setenv( "NCCL_DEBUG_SUBSYS", "ALL", 1);
    // setenv( "NCCL_ALGO", "Ring", 1 );

    // User allocate resources
    int total_data_size = 512*1024*1024;
    int num_comm = 2;
    std::vector<int> devs = {0,1};
    std::vector<ncclComm_t> comms( num_comm );
    std::vector<cudaStream_t> streams( num_comm );

    printf("User init comm\n");
    NCCLCHECK(ncclCommInitAll( comms.data(), num_comm, devs.data() ));

    printf("Get number of blnk+ helper\n");
    int num_helper;
    NCCLCHECK(blinkplusGetHelperCnt( comms.data(), num_comm, devs.data(), &num_helper ));

    printf("blink+ init comm and data on %d helper\n", num_helper );
    NCCLCHECK(blinkplusCommInitAll( comms.data(), num_comm, devs.data(), num_helper ));

    printf("User user stream data\n");
    for ( int i = 0; i < num_comm; ++i )
    {
        CUDACHECK(cudaSetDevice( devs[ i ] ));
        CUDACHECK(cudaStreamCreateWithFlags( &(streams[i]), cudaStreamNonBlocking ));
    }

    printf("Initial data of size %d\n", total_data_size);
    // currently ignore none dividable data case
    int chunk_data_size = total_data_size / (num_helper + 1);

    std::vector<int**> sendbuffs( num_helper + 1 );
    std::vector<int**> recvbuffs( num_helper + 1 );
    std::vector<int> chunk_data_sizes( chunk_data_size, (num_helper+1) );

    for ( int group_i = 0; group_i < sendbuffs.size(); ++group_i )
    {
      sendbuffs[ group_i ] = (int**)malloc( num_comm * sizeof(int*) );
      recvbuffs[ group_i ] = (int**)malloc( num_comm * sizeof(int*) );

      for ( int comm_i = 0; comm_i < num_comm; ++comm_i )
      {
        CUDACHECK(cudaMalloc( (sendbuffs[ group_i ] + comm_i), chunk_data_size * sizeof(int)));
        CUDACHECK(cudaMalloc( (recvbuffs[ group_i ] + comm_i), chunk_data_size * sizeof(int)));
        CUDACHECK(cudaMemset( sendbuffs[ group_i ][ comm_i ], 1, chunk_data_size * sizeof(int)));
        CUDACHECK(cudaMemset( recvbuffs[ group_i ][ comm_i ], 0, chunk_data_size * sizeof(int)));    
      }
    }
  
    printf("User run broadcast\n");
    NCCLCHECK(ncclGroupStart());
    for ( int i = 0; i < num_comm; ++i ) 
    {
        // [0] for user group
        NCCLCHECK(ncclBroadcast((const void*)sendbuffs[ 0 ][ i ], \
                                (void*)recvbuffs[ 0 ][ i ], \
                                chunk_data_size, ncclInt, 0, \
                                comms[i], \
                                streams[i]));
    }
    NCCLCHECK(ncclGroupEnd());

    printf("blink+ run broadcast\n");
    // +1 to displace over [0] for user group
    NCCLCHECK( blinkplusBroadcast( comms.data(), num_comm, devs.data(), \
      (const void***)(sendbuffs.data() + 1), (void***)(recvbuffs.data() + 1), \
      chunk_data_sizes.data(), ncclInt, 0 ) );

    printf("User run allreduce\n");
    NCCLCHECK(ncclGroupStart());
    for ( int i = 0; i < num_comm; ++i ) 
    {
        NCCLCHECK(ncclAllReduce((const void*)sendbuffs[ 0 ][ i ], \
                                (void*)recvbuffs[ 0 ][ i ], \
                                chunk_data_size, ncclInt, ncclSum, \
                                comms[i], \
                                streams[i]));
    }
    NCCLCHECK(ncclGroupEnd());

    printf("blink+ run allreduce\n");
     // +1 to displace over [0] for user group
    NCCLCHECK( blinkplusAllReduce( comms.data(), num_comm, devs.data(), \
      (const void***)(sendbuffs.data() + 1), (void***)(recvbuffs.data() + 1), \
      chunk_data_sizes.data(), ncclInt, ncclSum ) );

    printf("User sync stream\n");
    for ( int i = 0; i < num_comm; ++i ) 
    {
      CUDACHECK(cudaSetDevice( devs[i]));
      CUDACHECK(cudaStreamSynchronize( streams[i]));
    }

    printf("blink+ sync stream\n");
    NCCLCHECK( blinkplusStreamSynchronize( comms.data() ) );

    printf("User free buffer\n");
    for ( int group_i = 0; group_i < sendbuffs.size(); ++group_i )
    {
      for ( int comm_i = 0; comm_i < num_comm; ++comm_i )
      {
        CUDACHECK(cudaSetDevice( devs[ comm_i ] ));
        CUDACHECK(cudaFree( sendbuffs[ group_i ][ comm_i ] ));
        CUDACHECK(cudaFree( recvbuffs[ group_i ][ comm_i ] ));
      }
    }

    printf("User free comm\n");
    for ( int i = 0; i < num_comm; ++i ) 
    {
        ncclCommDestroy( comms[i]);
    }

    printf("blink+ free buffer and comm\n");
    NCCLCHECK( blinkplusCommDestroy( comms.data(), num_comm, devs.data() ) );

    printf("Success \n");
    return 0;
}