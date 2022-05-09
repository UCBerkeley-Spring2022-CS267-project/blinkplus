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
        printf("Usage ./time_nccl_broadcast_user03_3group 0 3 NUM_WARMUP NUM_ITER TOTAL_DATA_SIZE\n");
        exit(1);
    }

    setenv( "NCCL_PROTO", "Simple", 1);
    setenv( "NCCL_ALGO", "Tree", 1 );

    printf("%s:: NCCL Version %d.%d.%d\n", __func__, NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH );

    // User allocate resources
    int total_data_size = atoi( argv[5] )*1024*1024;
    int num_warmup = atoi( argv[3] );
    int num_iters = atoi( argv[4] );
  
    int user_group_num_comm = 2;
    std::vector<int> user_group_devs = { 0,3 };
    std::vector<ncclComm_t> user_group_comms( user_group_num_comm );
    std::vector<cudaStream_t> user_group_streams( user_group_num_comm );

    int helper_group_1_num_comm = 3;
    std::vector<int> helper_group_1_devs = { 0,1,3 };
    std::vector<ncclComm_t> helper_group_1_comms( helper_group_1_num_comm );
    std::vector<cudaStream_t> helper_group_1_streams( helper_group_1_num_comm );

    int helper_group_2_num_comm = 3;
    std::vector<int> helper_group_2_devs = { 0,2,3 };
    std::vector<ncclComm_t> helper_group_2_comms( helper_group_2_num_comm );
    std::vector<cudaStream_t> helper_group_2_streams( helper_group_2_num_comm );

    printf("%s:: User GPU %d, %d\n", __func__, user_group_devs[0], user_group_devs[1]);
    printf("%s:: Helper 1 GPU %d, %d, %d\n", __func__, helper_group_1_devs[0], helper_group_1_devs[1], helper_group_1_devs[2]);
    printf("%s:: Helper 2 GPU %d, %d, %d\n", __func__, helper_group_2_devs[0], helper_group_2_devs[1], helper_group_2_devs[2]);

    printf("%s:: Init stream data\n", __func__ );
    for ( int i = 0; i < user_group_num_comm; ++i )
    {
        CUDACHECK(cudaSetDevice( user_group_devs[ i ] ));
        CUDACHECK(cudaStreamCreate( &(user_group_streams[i]) ));
    }

    for ( int i = 0; i < helper_group_1_num_comm; ++i )
    {
        CUDACHECK(cudaSetDevice( helper_group_1_devs[ i ] ));
        CUDACHECK(cudaStreamCreate( &(helper_group_1_streams[i]) ));
    }

    for ( int i = 0; i < helper_group_2_num_comm; ++i )
    {
        CUDACHECK(cudaSetDevice( helper_group_2_devs[ i ] ));
        CUDACHECK(cudaStreamCreate( &(helper_group_2_streams[i]) ));
    }

    printf("=========%s:: Initial data of size %d MB=========\n", __func__, int(atoi(argv[5]) * sizeof(uint8_t)));
    // currently ignore none dividable data case
    int chunk_data_size = total_data_size / 4;

    uint8_t** user_group_sendbuff = (uint8_t**)malloc(user_group_num_comm * sizeof(uint8_t*));
    uint8_t** user_group_recvbuff = (uint8_t**)malloc(user_group_num_comm * sizeof(uint8_t*));

    uint8_t** helper_group_1_sendbuff = (uint8_t**)malloc(helper_group_1_num_comm * sizeof(uint8_t*));
    uint8_t** helper_group_1_recvbuff = (uint8_t**)malloc(helper_group_1_num_comm * sizeof(uint8_t*));

    uint8_t** helper_group_2_sendbuff = (uint8_t**)malloc(helper_group_2_num_comm * sizeof(uint8_t*));
    uint8_t** helper_group_2_recvbuff = (uint8_t**)malloc(helper_group_2_num_comm * sizeof(uint8_t*));

    std::vector<uint8_t> h_sendbuff( chunk_data_size * 2 );
    for ( int i = 0; i < chunk_data_size * 2; ++i )
    {
      h_sendbuff[ i ] = i;
    }

    for ( int comm_i = 0; comm_i < user_group_num_comm; ++comm_i )
    {
      CUDACHECK(cudaSetDevice( user_group_devs[ comm_i ] ));
      CUDACHECK(cudaMalloc( (user_group_sendbuff + comm_i), chunk_data_size * 2 * sizeof(uint8_t) ));
      CUDACHECK(cudaMalloc( (user_group_recvbuff + comm_i), chunk_data_size * 2 * sizeof(uint8_t)));
      CUDACHECK(cudaMemcpy( user_group_sendbuff[ comm_i ], h_sendbuff.data(), chunk_data_size * 2 * sizeof(uint8_t), cudaMemcpyHostToDevice ));
      CUDACHECK(cudaMemset( user_group_recvbuff[ comm_i ], 0, chunk_data_size * 2 * sizeof(uint8_t)));
    }

    for ( int comm_i = 0; comm_i < helper_group_1_num_comm; ++comm_i )
    {
      CUDACHECK(cudaSetDevice( helper_group_1_devs[ comm_i ] ));
      CUDACHECK(cudaMalloc( (helper_group_1_sendbuff + comm_i), chunk_data_size * sizeof(uint8_t) ));
      CUDACHECK(cudaMalloc( (helper_group_1_recvbuff + comm_i), chunk_data_size * sizeof(uint8_t)));
      CUDACHECK(cudaMemcpy( helper_group_1_sendbuff[ comm_i ], h_sendbuff.data(), chunk_data_size * sizeof(uint8_t), cudaMemcpyHostToDevice ));
      CUDACHECK(cudaMemset( helper_group_1_recvbuff[ comm_i ], 0, chunk_data_size * sizeof(uint8_t)));    
    }

    for ( int comm_i = 0; comm_i < helper_group_2_num_comm; ++comm_i )
    {
      CUDACHECK(cudaSetDevice( helper_group_2_devs[ comm_i ] ));
      CUDACHECK(cudaMalloc( (helper_group_2_sendbuff + comm_i), chunk_data_size * sizeof(uint8_t) ));
      CUDACHECK(cudaMalloc( (helper_group_2_recvbuff + comm_i), chunk_data_size * sizeof(uint8_t)));
      CUDACHECK(cudaMemcpy( helper_group_2_sendbuff[ comm_i ], h_sendbuff.data(), chunk_data_size * sizeof(uint8_t), cudaMemcpyHostToDevice ));
      CUDACHECK(cudaMemset( helper_group_2_recvbuff[ comm_i ], 0, chunk_data_size * sizeof(uint8_t)));    
    }

    NCCLCHECK(ncclCommInitAll( user_group_comms.data(), user_group_num_comm, user_group_devs.data() ));

    setenv( "NCCL_GRAPH_FILE", std::getenv("BLINKPLUS_GRAPH_FILE_CHAIN_013") , 1 );
    NCCLCHECK(ncclCommInitAll( helper_group_1_comms.data(), helper_group_1_num_comm, helper_group_1_devs.data() ));

    setenv( "NCCL_GRAPH_FILE", std::getenv("BLINKPLUS_GRAPH_FILE_CHAIN_023") , 1 );
    NCCLCHECK(ncclCommInitAll( helper_group_2_comms.data(), helper_group_2_num_comm, helper_group_2_devs.data() ));

    printf("=====Start WarmUp Iters: %d =====\n", num_warmup);
    for (int iter = 0; iter < num_warmup; iter++) 
    {
        NCCLCHECK(ncclGroupStart());

        NCCLCHECK(ncclGroupStart());
        for ( int i = 0; i < user_group_num_comm; ++i ) 
        {
            NCCLCHECK(ncclBroadcast((const void*)user_group_sendbuff[ i ], \
                                    (void*)user_group_recvbuff[ i ], \
                                    chunk_data_size * 2, ncclInt8, user_group_devs[ 0 ], \
                                    user_group_comms[i], \
                                    user_group_streams[i]));
        }
        NCCLCHECK(ncclGroupEnd());

        NCCLCHECK(ncclGroupStart());
        for ( int i = 0; i < helper_group_1_num_comm; ++i ) 
        {
            NCCLCHECK(ncclBroadcast((const void*)helper_group_1_sendbuff[ i ], \
                                    (void*)helper_group_1_recvbuff[ i ], \
                                    chunk_data_size, ncclInt8, helper_group_1_devs[ 0 ], \
                                    helper_group_1_comms[i], \
                                    helper_group_1_streams[i]));
        }
        NCCLCHECK(ncclGroupEnd());

        NCCLCHECK(ncclGroupStart());
        for ( int i = 0; i < helper_group_2_num_comm; ++i ) 
        {
            NCCLCHECK(ncclBroadcast((const void*)helper_group_2_sendbuff[ i ], \
                                    (void*)helper_group_2_recvbuff[ i ], \
                                    chunk_data_size, ncclInt8, helper_group_2_devs[ 0 ], \
                                    helper_group_2_comms[i], \
                                    helper_group_2_streams[i]));
        }
        NCCLCHECK(ncclGroupEnd());

        NCCLCHECK(ncclGroupEnd());
    }

    printf("%s:: Synchronize warmup\n", __func__ );
    for ( int i = 0; i < user_group_num_comm; ++i ) 
    {
      CUDACHECK(cudaSetDevice( user_group_devs[i]));
      CUDACHECK(cudaStreamSynchronize( user_group_streams[i] ));
    }

    for ( int i = 0; i < helper_group_1_num_comm; ++i ) 
    {
      CUDACHECK(cudaSetDevice( helper_group_1_devs[i] ));
      CUDACHECK(cudaStreamSynchronize( helper_group_1_streams[i] ));
    }

    for ( int i = 0; i < helper_group_2_num_comm; ++i ) 
    {
      CUDACHECK(cudaSetDevice( helper_group_2_devs[i] ));
      CUDACHECK(cudaStreamSynchronize( helper_group_2_streams[i] ));
    }

    printf("=====End WarmUp=====\n");

    // Start timing
    printf("=====Start Timing, Iters: %d ======\n", num_iters);
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < num_iters; iter++) 
    {
        NCCLCHECK(ncclGroupStart());

        NCCLCHECK(ncclGroupStart());
        for ( int i = 0; i < user_group_num_comm; ++i )
        {
            NCCLCHECK(ncclBroadcast((const void*)user_group_sendbuff[ i ], \
                                    (void*)user_group_recvbuff[ i ], \
                                    chunk_data_size * 2, ncclInt8, user_group_devs[ 0 ], \
                                    user_group_comms[i], \
                                    user_group_streams[i]));
        }
        NCCLCHECK(ncclGroupEnd());

        NCCLCHECK(ncclGroupStart());
        for ( int i = 0; i < helper_group_1_num_comm; ++i )
        {
            NCCLCHECK(ncclBroadcast((const void*)helper_group_1_sendbuff[ i ], \
                                    (void*)helper_group_1_recvbuff[ i ], \
                                    chunk_data_size, ncclInt8, helper_group_1_devs[ 0 ], \
                                    helper_group_1_comms[i], \
                                    helper_group_1_streams[i]));
        }
        NCCLCHECK(ncclGroupEnd());

        NCCLCHECK(ncclGroupStart());
        for ( int i = 0; i < helper_group_2_num_comm; ++i ) 
        {
            NCCLCHECK(ncclBroadcast((const void*)helper_group_2_sendbuff[ i ], \
                                    (void*)helper_group_2_recvbuff[ i ], \
                                    chunk_data_size, ncclInt8, helper_group_2_devs[ 0 ], \
                                    helper_group_2_comms[i], \
                                    helper_group_2_streams[i]));
        }
        NCCLCHECK(ncclGroupEnd());

        NCCLCHECK(ncclGroupEnd());
    }

    for ( int i = 0; i < user_group_num_comm; ++i ) 
    {
      CUDACHECK(cudaSetDevice( user_group_devs[i]));
      CUDACHECK(cudaStreamSynchronize( user_group_streams[i]));
    }

    // End timing
    {
      printf("=====End Timing User======\n");
      auto delta = std::chrono::high_resolution_clock::now() - start;
      double deltaSec = std::chrono::duration_cast<std::chrono::duration<double>>(delta).count();
      deltaSec = deltaSec / num_iters;
      double timeUsec = deltaSec*1.0E6;
      double bw = total_data_size / 4 * 2 * sizeof(uint8_t) / 1.0E9 / deltaSec;
      printf("%s:: User Average of %d Iters, data: %d MB,  Elapsed Time: %7.5f (us), BandWidth: %7.5f (GB/s)\n", \
                  __func__, num_iters, int(atoi(argv[5]) / 4 * 2 * sizeof(uint8_t)), timeUsec,  bw);  
    }

    for ( int i = 0; i < helper_group_1_num_comm; ++i ) 
    {
      CUDACHECK(cudaSetDevice( helper_group_1_devs[i] ));
      CUDACHECK(cudaStreamSynchronize( helper_group_1_streams[i] ));
    }

    for ( int i = 0; i < helper_group_2_num_comm; ++i ) 
    {
      CUDACHECK(cudaSetDevice( helper_group_2_devs[i] ));
      CUDACHECK(cudaStreamSynchronize( helper_group_2_streams[i] ));
    }

    // End timing
    {
      printf("=====End Timing (Total) Two Helper======\n");
      auto delta = std::chrono::high_resolution_clock::now() - start;
      double deltaSec = std::chrono::duration_cast<std::chrono::duration<double>>(delta).count();
      deltaSec = deltaSec / num_iters;
      double timeUsec = deltaSec*1.0E6;
      double bw = total_data_size * sizeof(uint8_t) / 1.0E9 / deltaSec;
      printf("%s:: Total Average of %d Iters, data: %d MB,  Elapsed Time: %7.5f (us), BandWidth: %7.5f (GB/s)\n", \
                  __func__, num_iters, int(atoi(argv[5]) * sizeof(uint8_t)), timeUsec,  bw);
    }

    printf("%s:: check data correctness after stream synchronize\n", __func__);
    std::vector<uint8_t> h_recvbuff( chunk_data_size * 2 );

    for ( int comm_i = 0; comm_i < user_group_num_comm; ++comm_i )
    {
      CUDACHECK( cudaMemcpy( h_recvbuff.data(), user_group_recvbuff[ comm_i ], chunk_data_size * 2 * sizeof( uint8_t ), cudaMemcpyDeviceToHost ));
      for ( int i = 0; i < h_recvbuff.size(); ++i )
      {
        if ( h_recvbuff[i] != h_sendbuff[i] )
        {
          printf("%s:: Check recv on user group comm %d failed, expected %d but have %d\n", __func__, comm_i, h_sendbuff[i], h_recvbuff[i] );
        }
      }
    }

    for ( int comm_i = 0; comm_i < helper_group_1_num_comm; ++comm_i )
    {
      CUDACHECK( cudaMemcpy( h_recvbuff.data(), helper_group_1_recvbuff[ comm_i ], chunk_data_size * sizeof( uint8_t ), cudaMemcpyDeviceToHost ));
      for ( int i = 0; i < h_recvbuff.size(); ++i )
      {
        if ( h_recvbuff[i] != h_sendbuff[i] )
        {
          printf("%s:: Check recv on helper group comm %d failed, expected %d but have %d\n", __func__, comm_i, h_sendbuff[i], h_recvbuff[i] );
        }
      }
    }

    for ( int comm_i = 0; comm_i < helper_group_2_num_comm; ++comm_i )
    {
      CUDACHECK( cudaMemcpy( h_recvbuff.data(), helper_group_2_recvbuff[ comm_i ], chunk_data_size * sizeof( uint8_t ), cudaMemcpyDeviceToHost ));
      for ( int i = 0; i < h_recvbuff.size(); ++i )
      {
        if ( h_recvbuff[i] != h_sendbuff[i] )
        {
          printf("%s:: Check recv on helper group comm %d failed, expected %d but have %d\n", __func__, comm_i, h_sendbuff[i], h_recvbuff[i] );
        }
      }
    }

    for ( int comm_i = 0; comm_i < user_group_num_comm; ++comm_i )
    {
      CUDACHECK(cudaSetDevice( user_group_devs[ comm_i ] ));
      CUDACHECK(cudaFree( user_group_sendbuff[ comm_i ] ));
      CUDACHECK(cudaFree( user_group_recvbuff[ comm_i ] ));
    }

    for ( int comm_i = 0; comm_i < helper_group_1_num_comm; ++comm_i )
    {
      CUDACHECK(cudaSetDevice( helper_group_1_devs[ comm_i ] ));
      CUDACHECK(cudaFree( helper_group_1_sendbuff[ comm_i ] ));
      CUDACHECK(cudaFree( helper_group_1_recvbuff[ comm_i ] ));
    }

    for ( int comm_i = 0; comm_i < helper_group_2_num_comm; ++comm_i )
    {
      CUDACHECK(cudaSetDevice( helper_group_2_devs[ comm_i ] ));
      CUDACHECK(cudaFree( helper_group_2_sendbuff[ comm_i ] ));
      CUDACHECK(cudaFree( helper_group_2_recvbuff[ comm_i ] ));
    }

    for ( int i = 0; i < user_group_num_comm; ++i ) 
    {
        ncclCommDestroy( user_group_comms[i]);
    }

    for ( int i = 0; i < helper_group_1_num_comm; ++i ) 
    {
        ncclCommDestroy( helper_group_1_comms[i]);
    }

    for ( int i = 0; i < helper_group_2_num_comm; ++i ) 
    {
        ncclCommDestroy( helper_group_2_comms[i]);
    }

    printf("%s:: Success \n", __func__);
    return 0;
}