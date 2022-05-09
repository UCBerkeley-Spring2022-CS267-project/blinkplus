#include <stdio.h>
#include <cstdlib>
#include <vector>
#include <chrono>
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


int main(int argc, char* argv[])
{
    if ( argc != 6 )
    {
        printf("Usage ./time_nccl_allreduce GU1 GPU2 NUM_WARMUP NUM_ITER TOTAL_DATA_SIZE\n");
        exit(1);
    }

    setenv( "NCCL_PROTO", "Simple", 1);
    setenv( "NCCL_ALGO", "Tree", 1 );

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

    printf("=========%s:: Initial data of size %d MB=========\n", __func__, int(atoi(argv[5]) * sizeof(uint8_t)));
    // currently ignore none dividable data case
    int chunk_data_size = total_data_size;

    std::vector<uint8_t**> sendbuffs( 1 );
    std::vector<uint8_t**> recvbuffs( 1 );

    std::vector<uint8_t> h_sendbuff( chunk_data_size );
    for ( int i = 0; i < chunk_data_size; ++i )
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
        CUDACHECK(cudaMalloc( (sendbuffs[ group_i ] + comm_i), chunk_data_size * sizeof(uint8_t) ));
        CUDACHECK(cudaMalloc( (recvbuffs[ group_i ] + comm_i), chunk_data_size * sizeof(uint8_t)));
        CUDACHECK(cudaMemcpy( sendbuffs[ group_i ][ comm_i ], h_sendbuff.data(), chunk_data_size * sizeof(uint8_t), cudaMemcpyHostToDevice ));
        CUDACHECK(cudaMemset( recvbuffs[ group_i ][ comm_i ], -1, chunk_data_size * sizeof(uint8_t)));    
      }
    }

    NCCLCHECK(ncclCommInitAll( comms.data(), num_comm, devs.data() ));

    printf("=====Start WarmUp Iters: %d =====\n", num_warmup);
    for (int iter = 0; iter < num_warmup; iter++) {
      NCCLCHECK(ncclGroupStart());
      for ( int i = 0; i < num_comm; ++i ) 
      {
          NCCLCHECK(ncclAllReduce((const void*)sendbuffs[ 0 ][ i ], \
                                  (void*)recvbuffs[ 0 ][ i ], \
                                  chunk_data_size, ncclInt8, ncclSum, \
                                  comms[i], \
                                  streams[i]));
      }
      NCCLCHECK(ncclGroupEnd());
    }

    printf("%s:: Synchronize warmup\n", __func__ );
    for ( int i = 0; i < num_comm; ++i ) 
    {
      CUDACHECK(cudaSetDevice( devs[i]));
      CUDACHECK(cudaStreamSynchronize( streams[i]));
    }
    printf("=====End WarmUp=====\n");

    // Start timing
    printf("=====Start Timing, Iters: %d ======\n", num_iters);
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < num_iters; iter++) {
      NCCLCHECK(ncclGroupStart());
      for ( int i = 0; i < num_comm; ++i ) 
      {
        NCCLCHECK(ncclAllReduce((const void*)sendbuffs[ 0 ][ i ], \
                                (void*)recvbuffs[ 0 ][ i ], \
                                chunk_data_size, ncclInt8, ncclSum, \
                                comms[i], \
                                streams[i]));
      }
      NCCLCHECK(ncclGroupEnd());
    }
    for ( int i = 0; i < num_comm; ++i ) 
    {
      CUDACHECK(cudaSetDevice( devs[i]));
      CUDACHECK(cudaStreamSynchronize( streams[i]));
    }
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
    std::vector<uint8_t> h_recvbuff( chunk_data_size );

    for ( int group_i = 0; group_i < sendbuffs.size(); ++group_i )
    {
      for ( int comm_i = 0; comm_i < num_comm; ++comm_i )
      {
        // Copy memory to cpu
        CUDACHECK( cudaMemcpy( h_recvbuff.data(), recvbuffs[ group_i ][ comm_i ], chunk_data_size * sizeof( uint8_t ), cudaMemcpyDeviceToHost ));
 
        // check result
        for ( int i = 0; i < h_recvbuff.size(); ++i )
        {
          if ( h_recvbuff[i] != 22 )
          {
            printf("%s:: Check recv on group %d, comm %d failed, expected %d but have %d\n", __func__, group_i, comm_i, 22, h_recvbuff[i] );
          }
        }
      }
    }

    for ( int group_i = 0; group_i < sendbuffs.size(); ++group_i )
    {
      for ( int comm_i = 0; comm_i < num_comm; ++comm_i )
      {
        CUDACHECK(cudaSetDevice( devs[ comm_i ] ));
        CUDACHECK(cudaFree( sendbuffs[ group_i ][ comm_i ] ));
        CUDACHECK(cudaFree( recvbuffs[ group_i ][ comm_i ] ));
      }
    }

    for ( int i = 0; i < num_comm; ++i ) 
    {
        ncclCommDestroy( comms[i]);
    }

    printf("%s:: Success \n", __func__);
    return 0;
}