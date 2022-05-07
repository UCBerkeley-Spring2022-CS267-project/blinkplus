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

int main(int argc, char* argv[])
{
    setenv( "NCCL_PROTO", "Simple", 1);
    setenv( "NCCL_DEBUG", "Info", 1);
    setenv( "NCCL_DEBUG_SUBSYS", "ALL", 1);
    setenv( "NCCL_ALGO", "Ring", 1 );

    printf("%s:: NCCL Version %d.%d.%d\n", __func__, NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH );

    // User allocate resources
    int total_data_size = 512; // 512*1024*1024;
    int num_comm = 2;
    std::vector<int> devs = { 0,1 };
    std::vector<ncclComm_t> comms( num_comm );
    std::vector<cudaStream_t> streams( num_comm );

    printf("%s:: User GPU %d, %d\n", __func__, devs[0], devs[1]);

    printf("%s:: User user stream data\n", __func__ );
    for ( int i = 0; i < num_comm; ++i )
    {
        CUDACHECK(cudaSetDevice( devs[ i ] ));
        CUDACHECK(cudaStreamCreate( &(streams[i]) ));
    }

    printf("%s:: Get number of blnk+ helper\n", __func__);
    int num_helper;
    NCCLCHECK(blinkplusGetHelperCnt( comms.data(), num_comm, devs.data(), &num_helper ));

    printf("%s:: Initial data of size %d\n", __func__, total_data_size);
    // currently ignore none dividable data case
    int chunk_data_size = total_data_size / (num_helper + 1);

    std::vector<int**> sendbuffs( num_helper + 1 );
    std::vector<int**> recvbuffs( num_helper + 1 );
    std::vector<int> chunk_data_sizes( chunk_data_size, (num_helper+1) );

    std::vector<int> h_sendbuff( chunk_data_size );
    for ( int i = 0; i < chunk_data_size; ++i )
    {
      h_sendbuff[ i ] = i;
    }

    for ( int group_i = 0; group_i < sendbuffs.size(); ++group_i )
    {
      sendbuffs[ group_i ] = (int**)malloc( num_comm * sizeof(int*) );
      recvbuffs[ group_i ] = (int**)malloc( num_comm * sizeof(int*) );

      for ( int comm_i = 0; comm_i < num_comm; ++comm_i )
      {
        CUDACHECK(cudaSetDevice( devs[ comm_i ] ));
        CUDACHECK(cudaMalloc( (sendbuffs[ group_i ] + comm_i), chunk_data_size * sizeof(int)));
        CUDACHECK(cudaMalloc( (recvbuffs[ group_i ] + comm_i), chunk_data_size * sizeof(int)));
        //CUDACHECK(cudaMemset( sendbuffs[ group_i ][ comm_i ], 100, chunk_data_size * sizeof(int)));
        CUDACHECK(cudaMemcpy(sendbuffs[ group_i ][ comm_i ], h_sendbuff.data(), chunk_data_size * sizeof(int), cudaMemcpyHostToDevice ));
        CUDACHECK(cudaMemset( recvbuffs[ group_i ][ comm_i ], -1, chunk_data_size * sizeof(int)));    
      }
    }

    printf("%s:: User init comm\n", __func__ );
    NCCLCHECK(ncclCommInitAll( comms.data(), num_comm, devs.data() ));

    printf("%s:: blink+ init comm and data on %d helper\n", __func__, num_helper );
    NCCLCHECK(blinkplusCommInitAll( comms.data(), num_comm, devs.data() ));

    printf("%s:: User run broadcast\n", __func__);
    NCCLCHECK(ncclGroupStart());
    for ( int i = 0; i < num_comm; ++i ) 
    {
        // sendbuffs[ 0 ] for user group
        // devs[ 0 ] choose 0th device as root
        NCCLCHECK(ncclBroadcast((const void*)sendbuffs[ 0 ][ i ], \
                                (void*)recvbuffs[ 0 ][ i ], \
                                chunk_data_size, ncclInt, devs[ 0 ], \
                                comms[i], \
                                streams[i]));
    }
    NCCLCHECK(ncclGroupEnd());

    printf("%s:: blink+ run broadcast\n", __func__);
    // +1 to displace over [0] for user group
    NCCLCHECK( blinkplusBroadcast( comms.data(), num_comm, devs.data(), \
      (const void***)(sendbuffs.data() + 1), (void***)(recvbuffs.data() + 1), \
      chunk_data_sizes.data(), ncclInt, devs[ 0 ] ) );

    // printf("%s:: User run allreduce\n");
    // NCCLCHECK(ncclGroupStart());
    // for ( int i = 0; i < num_comm; ++i ) 
    // {
    //     NCCLCHECK(ncclAllReduce((const void*)sendbuffs[ 0 ][ i ], \
    //                             (void*)recvbuffs[ 0 ][ i ], \
    //                             chunk_data_size, ncclInt, ncclSum, \
    //                             comms[i], \
    //                             streams[i]));
    // }
    // NCCLCHECK(ncclGroupEnd());

    // printf("%s:: blink+ run allreduce\n");
    //  // +1 to displace over [0] for user group
    // NCCLCHECK( blinkplusAllReduce( comms.data(), num_comm, devs.data(), \
    //   (const void***)(sendbuffs.data() + 1), (void***)(recvbuffs.data() + 1), \
    //   chunk_data_sizes.data(), ncclInt, ncclSum ) );

    printf("%s:: User sync stream\n", __func__);
    for ( int i = 0; i < num_comm; ++i ) 
    {
      CUDACHECK(cudaSetDevice( devs[i]));
      CUDACHECK(cudaStreamSynchronize( streams[i]));
    }

    printf("%s:: blink+ sync stream\n", __func__);
    NCCLCHECK( blinkplusStreamSynchronize( comms.data() ) );

    printf("%s:: check data correctness after stream synchronize\n", __func__);
    
    std::vector<int> h_recvbuff( chunk_data_size );

    for ( int group_i = 0; group_i < sendbuffs.size(); ++group_i )
    {
      if ( group_i != 0 )
        continue;

      for ( int comm_i = 0; comm_i < num_comm; ++comm_i )
      {
        // Copy memory to cpu
        CUDACHECK( cudaMemcpy( h_recvbuff.data(), recvbuffs[ group_i ][ comm_i ], chunk_data_size * sizeof( int ), cudaMemcpyDeviceToHost ));
 
        // check result
        for ( int i = 0; i < h_recvbuff.size(); ++i )
        {
          if ( h_recvbuff[i] != h_sendbuff[i] )
          {
            printf("%s:: Check recv on group %d, comm %d failed, expected %d but have %d\n", __func__, group_i, comm_i, h_sendbuff[i], h_recvbuff[i] );
          }
        }
      }
    }

    printf("%s:: User free buffer\n", __func__);
    for ( int group_i = 0; group_i < sendbuffs.size(); ++group_i )
    {
      for ( int comm_i = 0; comm_i < num_comm; ++comm_i )
      {
        CUDACHECK(cudaSetDevice( devs[ comm_i ] ));
        CUDACHECK(cudaFree( sendbuffs[ group_i ][ comm_i ] ));
        CUDACHECK(cudaFree( recvbuffs[ group_i ][ comm_i ] ));
      }
    }

    printf("%s:: User free comm\n", __func__);
    for ( int i = 0; i < num_comm; ++i ) 
    {
        ncclCommDestroy( comms[i]);
    }

    printf("%s:: blink+ free buffer and comm\n", __func__);
    NCCLCHECK( blinkplusCommDestroy( comms.data(), num_comm, devs.data() ) );

    printf("%s:: Success \n", __func__);
    return 0;
}