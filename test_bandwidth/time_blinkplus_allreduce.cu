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


int main(int argc, char* argv[])
{
    if ( argc != 6 )
    {
        printf("Usage ./run_blinkplus_broadcast.out GU1 GPU2 NUM_WARMUP NUM_ITER TOTAL_DATA_SIZE\n");
        exit(1);
    }

    printf("%s:: NCCL Version %d.%d.%d\n", __func__, NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH );

    // User allocate resources
    int total_data_size = atoi( argv[5] )*1024*1024;
    int num_warmup = atoi( argv[3] );
    int num_iters = atoi( argv[4] );
    int num_comm = 2;
    std::vector<int> devs = { atoi(argv[1]), atoi(argv[2]) };
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
    std::vector<int> data_partitions;
    NCCLCHECK(blinkplusGetHelperCnt( comms.data(), num_comm, devs.data(), data_partitions ));

    printf("=========%s:: Initial data of size %d MB=========\n", __func__, int(atoi(argv[5]) * sizeof(uint8_t)));
    // currently ignore none dividable data case
    int num_data_chunk = 0;
    for ( auto data_partition_i : data_partitions )
    {
      num_data_chunk += data_partition_i;
    }
    int chunk_data_size = total_data_size / num_data_chunk;

    std::vector<uint8_t**> sendbuffs( data_partitions.size() );
    std::vector<uint8_t**> recvbuffs( data_partitions.size() );
    std::vector<int> chunk_data_sizes( data_partitions.size(), chunk_data_size );

    std::vector<uint8_t> h_sendbuff( chunk_data_size * num_data_chunk );
    for ( int i = 0; i < chunk_data_size * num_data_chunk; ++i )
    {
      h_sendbuff[ i ] = 11;
    }

    for ( int group_i = 0; group_i < sendbuffs.size(); ++group_i )
    {
      sendbuffs[ group_i ] = (uint8_t**)malloc( num_comm * sizeof(uint8_t*) );
      recvbuffs[ group_i ] = (uint8_t**)malloc( num_comm * sizeof(uint8_t*) );

      for ( int comm_i = 0; comm_i < num_comm; ++comm_i )
      {
        CUDACHECK(cudaSetDevice( devs[ comm_i ] ));
        CUDACHECK(cudaMalloc( (sendbuffs[ group_i ] + comm_i), chunk_data_size * data_partitions.at( group_i ) * sizeof(uint8_t) ));
        CUDACHECK(cudaMalloc( (recvbuffs[ group_i ] + comm_i), chunk_data_size * data_partitions.at( group_i ) * sizeof(uint8_t)));
        CUDACHECK(cudaMemcpy( sendbuffs[ group_i ][ comm_i ], h_sendbuff.data(), chunk_data_size * data_partitions.at( group_i ) * sizeof(uint8_t), cudaMemcpyHostToDevice ));
        CUDACHECK(cudaMemset( recvbuffs[ group_i ][ comm_i ], -1, chunk_data_size * data_partitions.at( group_i ) * sizeof(uint8_t)));    
      }
    }

    //printf("%s:: User init comm\n", __func__ );
    NCCLCHECK(ncclCommInitAll( comms.data(), num_comm, devs.data() ));

    //printf("%s:: blink+ init comm and data on %d helper\n", __func__, num_helper );
    NCCLCHECK(blinkplusCommInitAll( comms.data(), num_comm, devs.data() ));

    printf("=====Start WarmUp Iters: %d =====\n", num_warmup);
    for (int iter = 0; iter < num_warmup; iter++) 
    {
      //printf("%s:: User run broadcast\n", __func__);
      NCCLCHECK(ncclGroupStart());
      for ( int i = 0; i < num_comm; ++i ) 
      {
          // sendbuffs[ 0 ] for user group
          // devs[ 0 ] choose 0th device as root
          NCCLCHECK(ncclAllReduce((const void*)sendbuffs[ 0 ][ i ], \
                                  (void*)recvbuffs[ 0 ][ i ], \
                                  chunk_data_size * data_partitions.at( 0 ), ncclInt8, ncclSum, \
                                  comms[i], \
                                  streams[i]));
      }
      NCCLCHECK(ncclGroupEnd());
      //printf("%s:: blink+ run broadcast\n", __func__);
      // +1 to displace over [0] for user group
      NCCLCHECK( blinkplusAllReduce( comms.data(), num_comm, devs.data(), \
        (const void***)(sendbuffs.data() + 1), (void***)(recvbuffs.data() + 1), \
        chunk_data_sizes.data(), ncclInt8, ncclSum ) );
    }
    printf("%s:: Synchronize warmup\n", __func__ );
    //printf("%s:: User sync stream\n", __func__);
    for ( int i = 0; i < num_comm; ++i ) 
    {
      CUDACHECK(cudaSetDevice( devs[i]));
      CUDACHECK(cudaStreamSynchronize( streams[i]));
    }
    //printf("%s:: blink+ sync stream\n", __func__);
    NCCLCHECK( blinkplusStreamSynchronize( comms.data() ) );

    printf("=====End WarmUp=====\n");

    // Start timing
    printf("=====Start Timing, Iters: %d ======\n", num_iters);
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < num_iters; iter++) {
      //printf("%s:: User run broadcast\n", __func__);
      NCCLCHECK(ncclGroupStart());
      for ( int i = 0; i < num_comm; ++i ) 
      {
          // sendbuffs[ 0 ] for user group
          // devs[ 0 ] choose 0th device as root
          NCCLCHECK(ncclAllReduce((const void*)sendbuffs[ 0 ][ i ], \
                                  (void*)recvbuffs[ 0 ][ i ], \
                                  chunk_data_size * data_partitions.at( 0 ), ncclInt8, ncclSum, \
                                  comms[i], \
                                  streams[i]));
      }
      NCCLCHECK(ncclGroupEnd());
      //printf("%s:: blink+ run broadcast\n", __func__);
      // +1 to displace over [0] for user group
      NCCLCHECK( blinkplusAllReduce( comms.data(), num_comm, devs.data(), \
        (const void***)(sendbuffs.data() + 1), (void***)(recvbuffs.data() + 1), \
        chunk_data_sizes.data(), ncclInt8, ncclSum ) );
    }
    //printf("%s:: User sync stream\n", __func__);
    for ( int i = 0; i < num_comm; ++i ) 
    {
      CUDACHECK(cudaSetDevice( devs[i]));
      CUDACHECK(cudaStreamSynchronize( streams[i]));
    }
    //printf("%s:: blink+ sync stream\n", __func__);
    NCCLCHECK( blinkplusStreamSynchronize( comms.data() ) );

    // End timing
    printf("=====End Timing======\n");
    auto delta = std::chrono::high_resolution_clock::now() - start;
    double deltaSec = std::chrono::duration_cast<std::chrono::duration<double>>(delta).count();
    deltaSec = deltaSec / num_iters;
    double timeUsec = deltaSec*1.0E6;
    double bw = total_data_size * sizeof(uint8_t) / 1.0E9 / deltaSec;
    printf("%s:: Average of %d Iters, data: %d MB,  Elapsed Time: %7.5f (us), BandWidth: %7.5f (GB/s)\n", \
                __func__, num_iters, int(atoi(argv[5]) * sizeof(uint8_t)), timeUsec,  bw);

    printf("%s:: check data correctness after stream synchronize\n", __func__);
    std::vector<uint8_t> h_recvbuff( chunk_data_size * num_data_chunk );

    for ( int group_i = 0; group_i < sendbuffs.size(); ++group_i )
    {
      for ( int comm_i = 0; comm_i < num_comm; ++comm_i )
      {
        // Copy memory to cpu
        CUDACHECK( cudaMemcpy( h_recvbuff.data(), recvbuffs[ group_i ][ comm_i ], chunk_data_size * data_partitions.at( group_i ) * sizeof( uint8_t ), cudaMemcpyDeviceToHost ));
 
        // check result
        for ( int i = 0; i < chunk_data_size * data_partitions.at( group_i ); ++i )
        {
          if ( h_recvbuff[i] != 22 )
          {
            printf("%s:: Check recv on group %d, comm %d failed, expected %d but have %d\n", \
              __func__, group_i, comm_i, 22, h_recvbuff[i] );
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