#include <stdio.h>
#include <array>
#include <string>
#include <cstdlib>
#include <vector>
#include <cstdlib>
#include <string>
#include <chrono>
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

uint64_t getTime() 
{
  return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

int main(int argc, char* argv[])
{
    // setenv( "NCCL_PROTO", "Simple", 1);
    // setenv( "NCCL_DEBUG", "Info", 1);
    // setenv( "NCCL_DEBUG_SUBSYS", "ALL", 1);
    // setenv( "NCCL_ALGO", "Ring", 1 );

    printf("%s:: NCCL Version %d.%d.%d\n", __func__, NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH );

    // User allocate resources
    int total_data_size = atoi(argv[4])*1024*1024;
    int warm_up_iters = 5;
    int iters = atoi(argv[3]);
    int num_comm = 2;
    std::vector<int> devs = {atoi(argv[1]), atoi(argv[2])};
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

    printf("%d \n", atoi(argv[4]));
    printf("=========%s:: Initial data of size %d MB=========\n", __func__, int(atoi(argv[4]) * 4));
    // currently ignore none dividable data case
    int chunk_data_size = total_data_size / (num_helper + 1);

    std::vector<int**> sendbuffs( num_helper + 1 );
    std::vector<int**> recvbuffs( num_helper + 1 );
    std::vector<int> chunk_data_sizes( chunk_data_size, (num_helper+1) );

    std::vector<int> h_sendbuff( chunk_data_size );
    // for ( int i = 0; i < chunk_data_size; ++i )
    // {
    //   h_sendbuff[ i ] = 21;
    // }
    int right_ans = 47;   // 21  + 26  =  47
    printf("all reduce correct ans: %d \n", right_ans);
    for ( int group_i = 0; group_i < sendbuffs.size(); ++group_i )
    {
      sendbuffs[ group_i ] = (int**)malloc( num_comm * sizeof(int*) );
      recvbuffs[ group_i ] = (int**)malloc( num_comm * sizeof(int*) );

      for ( int comm_i = 0; comm_i < num_comm; ++comm_i )
      {
        for ( int j = 0; j < chunk_data_size; ++j )
        {
          h_sendbuff[ j ] = 21 + comm_i * 5;   // 21  + 26  =  47
        }
        CUDACHECK(cudaSetDevice( devs[ comm_i ] ));
        CUDACHECK(cudaMalloc( (sendbuffs[ group_i ] + comm_i), chunk_data_size * sizeof(int) ));
        CUDACHECK(cudaMalloc( (recvbuffs[ group_i ] + comm_i), chunk_data_size * sizeof(int)));
        //CUDACHECK(cudaMemset( sendbuffs[ group_i ][ comm_i ], 100, chunk_data_size * sizeof(int)));
        CUDACHECK(cudaMemcpy(sendbuffs[ group_i ][ comm_i ], h_sendbuff.data(), chunk_data_size * sizeof(int), cudaMemcpyHostToDevice ));
        CUDACHECK(cudaMemset( recvbuffs[ group_i ][ comm_i ], -1, chunk_data_size * sizeof(int)));    
      }
    }

    //printf("%s:: User init comm\n", __func__ );
    NCCLCHECK(ncclCommInitAll( comms.data(), num_comm, devs.data() ));

    //printf("%s:: blink+ init comm and data on %d helper\n", __func__, num_helper );
    NCCLCHECK(blinkplusCommInitAll( comms.data(), num_comm, devs.data() ));


    printf("=====Start WarmUp Iters: %d =====\n", warm_up_iters);
    for (int iter = 0; iter < warm_up_iters; iter++) {
      //printf("%s:: User run allreduce\n");
      NCCLCHECK(ncclGroupStart());
      for ( int i = 0; i < num_comm; ++i ) 
      {
          // sendbuffs[ 0 ] for user group
          // devs[ 0 ] choose 0th device as root
          NCCLCHECK(ncclAllReduce((const void*)sendbuffs[ 0 ][ i ], \
                                    (void*)recvbuffs[ 0 ][ i ], \
                                    chunk_data_size, ncclInt, ncclSum, \
                                    comms[i], \
                                    streams[i]));
      }
      NCCLCHECK(ncclGroupEnd());
      //printf("%s:: blink+ run allreduce\n");
      // +1 to displace over [0] for user group
      NCCLCHECK( blinkplusAllReduce( comms.data(), num_comm, devs.data(), \
        (const void***)(sendbuffs.data() + 1), (void***)(recvbuffs.data() + 1), \
        chunk_data_sizes.data(), ncclInt, ncclSum ) );

      //printf("%s:: User sync stream\n", __func__);
      for ( int i = 0; i < num_comm; ++i ) 
      {
        CUDACHECK(cudaSetDevice( devs[i]));
        CUDACHECK(cudaStreamSynchronize( streams[i]));
      }
      //printf("%s:: blink+ sync stream\n", __func__);
      NCCLCHECK( blinkplusStreamSynchronize( comms.data() ) );
    }
    printf("=====End WarmUp=====\n");

    //=========================================================================//
    // Start timing
    printf("=====Start Timing, Iters: %d ======\n", iters);
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iters; iter++) {
      //printf("%s:: User run allreduce\n", __func__);
      NCCLCHECK(ncclGroupStart());
      for ( int i = 0; i < num_comm; ++i ) 
      {
          // sendbuffs[ 0 ] for user group
          // devs[ 0 ] choose 0th device as root
          NCCLCHECK(ncclAllReduce((const void*)sendbuffs[ 0 ][ i ], \
                                    (void*)recvbuffs[ 0 ][ i ], \
                                    chunk_data_size, ncclInt, ncclSum, \
                                    comms[i], \
                                    streams[i]));
      }
      NCCLCHECK(ncclGroupEnd());
      //printf("%s:: blink+ run allreduce\n", __func__);
      // +1 to displace over [0] for user group
      NCCLCHECK( blinkplusAllReduce( comms.data(), num_comm, devs.data(), \
        (const void***)(sendbuffs.data() + 1), (void***)(recvbuffs.data() + 1), \
        chunk_data_sizes.data(), ncclInt, ncclSum ) );

      //printf("%s:: User sync stream\n", __func__);
      for ( int i = 0; i < num_comm; ++i ) 
      {
        CUDACHECK(cudaSetDevice( devs[i]));
        CUDACHECK(cudaStreamSynchronize( streams[i]));
      }
      //printf("%s:: blink+ sync stream\n", __func__);
      NCCLCHECK( blinkplusStreamSynchronize( comms.data() ) );
    }

    // End timing
    printf("=====End Timing======\n");
    auto delta = std::chrono::high_resolution_clock::now() - start;
    double deltaSec = std::chrono::duration_cast<std::chrono::duration<double>>(delta).count();
    deltaSec = deltaSec / iters;
    double timeUsec = deltaSec*1.0E6;
    double bw = total_data_size * sizeof(int) / 1.0E9 / deltaSec;
    printf("%s:: Average of %d Iters, data: %d MB,  Elapsed Time: %7.5f (us), BandWidth: %7.5f (GB/s)\n", \
                __func__, iters, int(atoi(argv[4]) * 4), timeUsec,  bw);

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
          if ( h_recvbuff[i] != right_ans )
          {
            printf("%s:: Check recv on group %d, comm %d failed, expected %d but have %d\n", __func__, group_i, comm_i, h_sendbuff[i], h_recvbuff[i] );
          }
        }
      }
    }

    //printf("%s:: User free buffer\n", __func__);
    for ( int group_i = 0; group_i < sendbuffs.size(); ++group_i )
    {
      for ( int comm_i = 0; comm_i < num_comm; ++comm_i )
      {
        CUDACHECK(cudaSetDevice( devs[ comm_i ] ));
        CUDACHECK(cudaFree( sendbuffs[ group_i ][ comm_i ] ));
        CUDACHECK(cudaFree( recvbuffs[ group_i ][ comm_i ] ));
      }
    }

    //printf("%s:: User free comm\n", __func__);
    for ( int i = 0; i < num_comm; ++i ) 
    {
        ncclCommDestroy( comms[i]);
    }

    //printf("%s:: blink+ free buffer and comm\n", __func__);
    NCCLCHECK( blinkplusCommDestroy( comms.data(), num_comm, devs.data() ) );

    printf("%s:: Success \n", __func__);
    return 0;
}