#include <stdio.h>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <string>
#include <stdexcept> // std::runtime_error
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


struct blinkplusHelperGroup
{
    std::vector<int> devs;
    std::vector<ncclComm_t> comms;
    std::vector<uint8_t*> sendbuff;
    std::vector<uint8_t*> recvbuff;
    std::vector<cudaStream_t> streams;
    std::string graph_filepath;
    int num_comms;

    blinkplusHelperGroup( const char* graph_filepath_cstr, std::vector<int> devs )
    {
        if ( std::getenv(graph_filepath_cstr) == nullptr )
        {
            throw std::runtime_error( std::string(graph_filepath_cstr) + " unset\n");
        }
        this->graph_filepath = std::getenv(graph_filepath_cstr);
        this->devs = devs;
        this->comms.resize( this->devs.size() );
        this->sendbuff.resize( this->devs.size() );
        this->recvbuff.resize( this->devs.size() );
        this->streams.resize( this->devs.size() );
        this->num_comms = this->devs.size();
    }
};


std::vector<blinkplusHelperGroup> blinkplusHelperGroupsContainer;


ncclResult_t blinkplusStreamSynchronize( )
{
  for ( int group_i = 0; group_i < blinkplusHelperGroupsContainer.size(); ++group_i )
  {  
    for ( int comm_j = 0; comm_j < blinkplusHelperGroupsContainer.at( group_i ).num_comms; comm_j++ )
    {
      #ifndef NDEBUG
      printf("%s:: group %d, comm %d call stream sync on dev %d\n", \
        __func__, group_i, comm_j, blinkplusHelperGroupsContainer.at( group_i ).devs.at( comm_j ) );
      #endif

      CUDACHECK(cudaSetDevice( blinkplusHelperGroupsContainer.at( group_i ).devs.at( comm_j ) ));
      CUDACHECK(cudaStreamSynchronize( blinkplusHelperGroupsContainer.at( group_i ).streams.at( comm_j ) ));
    }
  }
  return ncclSuccess;
}


int main(int argc, char* argv[])
{
    if ( argc != 6 )
    {
        printf("Usage ./time_nccl_broadcast_2group 0 1 NUM_WARMUP NUM_ITER TOTAL_DATA_SIZE\n");
        exit(1);
    }

    setenv( "NCCL_PROTO", "Simple", 1);
    setenv( "NCCL_ALGO", "Tree", 1 );

    printf("%s:: NCCL Version %d.%d.%d\n", __func__, NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH );

    // User allocate resources
    int total_data_size = atoi( argv[5] )*1024*1024;
    int num_warmup = atoi( argv[3] );
    int num_iters = atoi( argv[4] );
  
    blinkplusHelperGroupsContainer.clear();
    blinkplusHelperGroupsContainer.reserve(2);

    blinkplusHelperGroup user_group( "BLINKPLUS_GRAPH_FILE_CHAIN_01", std::vector<int>{0,1});
    blinkplusHelperGroupsContainer.emplace_back( "BLINKPLUS_GRAPH_FILE_CHAIN_021", std::vector<int>{0,2,1} );
    blinkplusHelperGroupsContainer.emplace_back( "BLINKPLUS_GRAPH_FILE_CHAIN_031", std::vector<int>{0,3,1} );

    printf("%s:: User GPU %d, %d\n", __func__, user_group.devs[0], user_group.devs[1]);
    printf("%s:: Helper 1 GPU %d, %d, %d\n", __func__, blinkplusHelperGroupsContainer[ 0 ].devs[0], \
                                                       blinkplusHelperGroupsContainer[ 0 ].devs[1], \
                                                       blinkplusHelperGroupsContainer[ 0 ].devs[2]);
    printf("%s:: Helper 2 GPU %d, %d, %d\n", __func__, blinkplusHelperGroupsContainer[ 1 ].devs[0], \
                                                       blinkplusHelperGroupsContainer[ 1 ].devs[1], \
                                                       blinkplusHelperGroupsContainer[ 1 ].devs[2]);

    printf("%s:: Init stream data\n", __func__ );
    for ( int i = 0; i < user_group.num_comms; ++i )
    {
        CUDACHECK(cudaSetDevice( user_group.devs[ i ] ));
        CUDACHECK(cudaStreamCreate( &(user_group.streams[i]) ));
    }

    for ( int group_i = 0; group_i < blinkplusHelperGroupsContainer.size(); ++group_i )
    {
        for ( int i = 0; i < blinkplusHelperGroupsContainer.at( group_i ).num_comms; ++i )
        {
            CUDACHECK(cudaSetDevice( blinkplusHelperGroupsContainer.at( group_i ).devs[ i ] ));
            CUDACHECK(cudaStreamCreate( &(blinkplusHelperGroupsContainer.at( group_i ).streams[i]) ));
        }
    }

    printf("=========%s:: Initial data of size %d MB=========\n", __func__, int(atoi(argv[5]) * sizeof(uint8_t)));
    // currently ignore none dividable data case
    int chunk_data_size = total_data_size / 3;

    std::vector<uint8_t> h_sendbuff( chunk_data_size );
    for ( int i = 0; i < chunk_data_size; ++i )
    {
      h_sendbuff[ i ] = i;
    }

    for ( int comm_i = 0; comm_i < user_group.num_comms; ++comm_i )
    {
      CUDACHECK(cudaSetDevice( user_group.devs[ comm_i ] ));
      CUDACHECK(cudaMalloc( (user_group.sendbuff.data() + comm_i), chunk_data_size * sizeof(uint8_t) ));
      CUDACHECK(cudaMalloc( (user_group.recvbuff.data() + comm_i), chunk_data_size * sizeof(uint8_t)));
      CUDACHECK(cudaMemcpy( user_group.sendbuff[ comm_i ], h_sendbuff.data(), chunk_data_size * sizeof(uint8_t), cudaMemcpyHostToDevice ));
      CUDACHECK(cudaMemset( user_group.recvbuff[ comm_i ], 0, chunk_data_size * sizeof(uint8_t)));    
    }

    // Allocate buffer for user group
    

    for ( int group_i = 0; group_i < blinkplusHelperGroupsContainer.size(); ++group_i )
    {
        for ( int comm_i = 0; comm_i < blinkplusHelperGroupsContainer.at( group_i ).num_comms; ++comm_i )
        {
        CUDACHECK(cudaSetDevice( blinkplusHelperGroupsContainer.at( group_i ).devs[ comm_i ] ));
        CUDACHECK(cudaMalloc( (blinkplusHelperGroupsContainer.at( group_i ).sendbuff.data() + comm_i), chunk_data_size * sizeof(uint8_t) ));
        CUDACHECK(cudaMalloc( (blinkplusHelperGroupsContainer.at( group_i ).recvbuff.data() + comm_i), chunk_data_size * sizeof(uint8_t)));
        CUDACHECK(cudaMemcpy( blinkplusHelperGroupsContainer.at( group_i ).sendbuff[ comm_i ], h_sendbuff.data(), chunk_data_size * sizeof(uint8_t), cudaMemcpyHostToDevice ));
        CUDACHECK(cudaMemset( blinkplusHelperGroupsContainer.at( group_i ).recvbuff[ comm_i ], 0, chunk_data_size * sizeof(uint8_t)));    
        }        
    }

    setenv( "NCCL_GRAPH_FILE", std::getenv("BLINKPLUS_GRAPH_FILE_CHAIN_01") , 1 );
    NCCLCHECK(ncclCommInitAll( user_group.comms.data(), user_group.num_comms, user_group.devs.data() ));

    for ( int group_i = 0; group_i < blinkplusHelperGroupsContainer.size(); ++group_i )
    {
        setenv( "NCCL_GRAPH_FILE", blinkplusHelperGroupsContainer.at( group_i ).graph_filepath.c_str() , 1 );
        NCCLCHECK(ncclCommInitAll( blinkplusHelperGroupsContainer.at( group_i ).comms.data(), \
                                   blinkplusHelperGroupsContainer.at( group_i ).num_comms, \
                                   blinkplusHelperGroupsContainer.at( group_i ).devs.data() ));
    }

    printf("=====Start WarmUp Iters: %d =====\n", num_warmup);
    for (int iter = 0; iter < num_warmup; iter++) 
    {
      NCCLCHECK(ncclGroupStart());
      for ( int i = 0; i < user_group.num_comms; ++i ) 
      {
          NCCLCHECK(ncclBroadcast((const void*)user_group.sendbuff[ i ], \
                                  (void*)user_group.recvbuff[ i ], \
                                  chunk_data_size, ncclInt8, user_group.devs[ 0 ], \
                                  user_group.comms[i], \
                                  user_group.streams[i]));
      }
      NCCLCHECK(ncclGroupEnd());

      for ( int group_i = 0; group_i < blinkplusHelperGroupsContainer.size(); ++group_i )
      {
        NCCLCHECK(ncclGroupStart());
        for ( int i = 0; i < blinkplusHelperGroupsContainer.at( group_i).num_comms; ++i ) 
        {
            NCCLCHECK(ncclBroadcast((const void*)blinkplusHelperGroupsContainer.at( group_i).sendbuff[ i ], \
                                    (void*)blinkplusHelperGroupsContainer.at( group_i).recvbuff[ i ], \
                                    chunk_data_size, ncclInt8, blinkplusHelperGroupsContainer.at( group_i).devs[ 0 ], \
                                    blinkplusHelperGroupsContainer.at( group_i).comms[i], \
                                    blinkplusHelperGroupsContainer.at( group_i).streams[i]));
        }
        NCCLCHECK(ncclGroupEnd());
      }
    }

    for ( int i = 0; i < user_group.num_comms; ++i ) 
    {
      CUDACHECK(cudaSetDevice( user_group.devs[i]));
      CUDACHECK(cudaStreamSynchronize( user_group.streams[i] ));
    }

    for ( int group_i = 0; group_i < blinkplusHelperGroupsContainer.size(); ++group_i )
    {
        for ( int i = 0; i < blinkplusHelperGroupsContainer.at( group_i ).num_comms; ++i ) 
        {
          CUDACHECK(cudaSetDevice( blinkplusHelperGroupsContainer.at( group_i ).devs[i] ));
          CUDACHECK(cudaStreamSynchronize( blinkplusHelperGroupsContainer.at( group_i ).streams[i] ));
        }    
    }

    printf("=====End WarmUp=====\n");

    // Start timing
    printf("=====Start Timing, Iters: %d ======\n", num_iters);
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < num_iters; iter++) 
    {
      NCCLCHECK(ncclGroupStart());
      for ( int i = 0; i < user_group.num_comms; ++i ) 
      {
          NCCLCHECK(ncclBroadcast((const void*)user_group.sendbuff[ i ], \
                                  (void*)user_group.recvbuff[ i ], \
                                  chunk_data_size, ncclInt8, user_group.devs[ 0 ], \
                                  user_group.comms[i], \
                                  user_group.streams[i]));
      }
      NCCLCHECK(ncclGroupEnd());

      for ( int group_i = 0; group_i < blinkplusHelperGroupsContainer.size(); ++group_i )
      {
        NCCLCHECK(ncclGroupStart());
        for ( int i = 0; i < blinkplusHelperGroupsContainer.at( group_i).num_comms; ++i ) 
        {
            NCCLCHECK(ncclBroadcast((const void*)blinkplusHelperGroupsContainer.at( group_i).sendbuff[ i ], \
                                    (void*)blinkplusHelperGroupsContainer.at( group_i).recvbuff[ i ], \
                                    chunk_data_size, ncclInt8, blinkplusHelperGroupsContainer.at( group_i).devs[ 0 ], \
                                    blinkplusHelperGroupsContainer.at( group_i).comms[i], \
                                    blinkplusHelperGroupsContainer.at( group_i).streams[i]));
        }
        NCCLCHECK(ncclGroupEnd());
      }
    }

    for ( int i = 0; i < user_group.num_comms; ++i ) 
    {
      CUDACHECK(cudaSetDevice( user_group.devs[i]));
      CUDACHECK(cudaStreamSynchronize( user_group.streams[i] ));
    }

    NCCLCHECK( blinkplusStreamSynchronize() );

    // End timing
    {
      printf("=====End Timing======\n");
      auto delta = std::chrono::high_resolution_clock::now() - start;
      double deltaSec = std::chrono::duration_cast<std::chrono::duration<double>>(delta).count();
      deltaSec = deltaSec / num_iters;
      double timeUsec = deltaSec*1.0E6;
      double bw = total_data_size * sizeof(uint8_t) / 1.0E9 / deltaSec;
      printf("%s:: Average of %d Iters, data: %d MB,  Elapsed Time: %7.5f (us), BandWidth: %7.5f (GB/s)\n", \
                  __func__, num_iters, int(atoi(argv[5]) * sizeof(uint8_t)), timeUsec,  bw);
    }

    printf("%s:: check data correctness after stream synchronize\n", __func__);
    std::vector<uint8_t> h_recvbuff( chunk_data_size );

    for ( int comm_i = 0; comm_i < user_group.num_comms; ++comm_i )
    {
      CUDACHECK( cudaMemcpy( h_recvbuff.data(), user_group.recvbuff[ comm_i ], chunk_data_size * sizeof( uint8_t ), cudaMemcpyDeviceToHost ));
      for ( int i = 0; i < h_recvbuff.size(); ++i )
      {
        if ( h_recvbuff[i] != h_sendbuff[i] )
        {
          printf("%s:: Check recv on user group comm %d failed, expected %d but have %d\n", __func__, comm_i, h_sendbuff[i], h_recvbuff[i] );
        }
      }
    }

    for ( int group_i = 0; group_i < blinkplusHelperGroupsContainer.size(); ++group_i )
    {
        for ( int comm_i = 0; comm_i < blinkplusHelperGroupsContainer.at( group_i ).num_comms; ++comm_i )
        {
            CUDACHECK( cudaMemcpy( h_recvbuff.data(), blinkplusHelperGroupsContainer.at( group_i ).recvbuff[ comm_i ], chunk_data_size * sizeof( uint8_t ), cudaMemcpyDeviceToHost ));
            for ( int i = 0; i < h_recvbuff.size(); ++i )
            {
                if ( h_recvbuff[i] != h_sendbuff[i] )
                {
                printf("%s:: Check recv on helper group comm %d failed, expected %d but have %d\n", __func__, comm_i, h_sendbuff[i], h_recvbuff[i] );
                }
            }
        }        
    }

    for ( int comm_i = 0; comm_i < user_group.num_comms; ++comm_i )
    {
      CUDACHECK(cudaSetDevice( user_group.devs[ comm_i ] ));
      CUDACHECK(cudaFree( user_group.sendbuff[ comm_i ] ));
      CUDACHECK(cudaFree( user_group.recvbuff[ comm_i ] ));
    }

    for ( int comm_i = 0; comm_i < user_group.num_comms; ++comm_i ) 
    {
        NCCLCHECK( ncclCommDestroy( user_group.comms[ comm_i] ) );
    }

    for ( int group_i = 0; group_i < blinkplusHelperGroupsContainer.size(); ++group_i )
    {
        for ( int comm_i = 0; comm_i < blinkplusHelperGroupsContainer.at( group_i ).num_comms; ++comm_i )
        {
            CUDACHECK(cudaSetDevice( blinkplusHelperGroupsContainer.at( group_i ).devs[ comm_i ] ));
            CUDACHECK(cudaFree( blinkplusHelperGroupsContainer.at( group_i ).sendbuff[ comm_i ] ));
            CUDACHECK(cudaFree( blinkplusHelperGroupsContainer.at( group_i ).recvbuff[ comm_i ] ));
            NCCLCHECK( ncclCommDestroy( blinkplusHelperGroupsContainer.at( group_i ).comms[ comm_i ] ) );
        }
    }


    printf("%s:: Success \n", __func__);
    return 0;
}