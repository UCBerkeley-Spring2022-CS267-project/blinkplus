
#include <string>
#include <map>
#include <vector>
#include <cstdio> // printf
#include <cstdlib> // std::getenv
#include <utility> // std::make_pair
#include <stdexcept> // std::runtime_error
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

  #define BLINKPLUS_BUFFER_SIZE_BYTES 100000000 // 1GB

struct blinkplusHelperGroup
{
    std::vector<int> devs;
    std::vector<ncclComm_t> comms;
    std::vector<int*> sendbuff;
    std::vector<int*> recvbuff;
    std::vector<cudaStream_t> streams;
    std::string graph_filepath;
    int num_comms;

    blinkplusHelperGroup( const char* graph_filepath_cstr, std::vector<int> devs )
    {
        if ( std::getenv(graph_filepath_cstr) == nullptr )
        {
            throw std::runtime_error("BLINKPLUS_GRAPH_FILE_CHAIN_XXX unset\n");
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


enum class blinkplusUserGroupType
{
  GROUP01,
  GROUP12,
  GROUP03,
  GROUP23,
  GROUP02,
  GROUP13,
};


std::vector<blinkplusHelperGroup> blinkplusHelperGroupsContainer;
blinkplusUserGroupType userGroupType;

std::map<blinkplusUserGroupType, std::vector<std::pair<std::string, std::vector<int>>>> configs
{
  { blinkplusUserGroupType::GROUP01, 
    std::vector<std::pair<std::string, std::vector<int>>>{
      std::make_pair( "BLINKPLUS_GRAPH_FILE_CHAIN_021", std::vector<int>{0,2,1} ),
      std::make_pair( "BLINKPLUS_GRAPH_FILE_CHAIN_031", std::vector<int>{0,3,1} ),
      std::make_pair( "BLINKPLUS_GRAPH_FILE_CHAIN_0321", std::vector<int>{0,3,2,1} ) } },
  { blinkplusUserGroupType::GROUP12, 
    std::vector<std::pair<std::string, std::vector<int>>>{
      std::make_pair( "BLINKPLUS_GRAPH_FILE_CHAIN_102", std::vector<int>{1,0,2} ),
      std::make_pair( "BLINKPLUS_GRAPH_FILE_CHAIN_132", std::vector<int>{1,3,2} ) } },
  { blinkplusUserGroupType::GROUP03,
      std::vector<std::pair<std::string, std::vector<int>>>{
        std::make_pair( "BLINKPLUS_GRAPH_FILE_CHAIN_013", std::vector<int>{0,1,3} ),
        std::make_pair( "BLINKPLUS_GRAPH_FILE_CHAIN_023", std::vector<int>{0,2,3} ) } },
  { blinkplusUserGroupType::GROUP23,
    std::vector<std::pair<std::string, std::vector<int>>>{
      std::make_pair( "BLINKPLUS_GRAPH_FILE_CHAIN_203", std::vector<int>{2,0,3} ),
      std::make_pair( "BLINKPLUS_GRAPH_FILE_CHAIN_213", std::vector<int>{2,1,3} ),
      std::make_pair( "BLINKPLUS_GRAPH_FILE_CHAIN_2103", std::vector<int>{2,1,0,3} ) } },
  { blinkplusUserGroupType::GROUP02,
    std::vector<std::pair<std::string, std::vector<int>>>{
      std::make_pair( "BLINKPLUS_GRAPH_FILE_CHAIN_032", std::vector<int>{0,3,2} ),
      std::make_pair( "BLINKPLUS_GRAPH_FILE_CHAIN_012", std::vector<int>{0,1,2} ) } },
  { blinkplusUserGroupType::GROUP13,
    std::vector<std::pair<std::string, std::vector<int>>>{
      std::make_pair( "BLINKPLUS_GRAPH_FILE_CHAIN_103", std::vector<int>{1,0,3} ),
      std::make_pair( "BLINKPLUS_GRAPH_FILE_CHAIN_123", std::vector<int>{1,2,3} ) } },
};

// NCCL_API(ncclResult_t, blinkplusGetHelperCnt, ncclComm_t* comms, int ndev, const int *devlist, int* helper_cnt );
ncclResult_t blinkplusGetHelperCnt( ncclComm_t* comm, int ndev, const int *devlist, int* helper_cnt )
{
  if ( ndev != 2 )
  {
    throw std::runtime_error("BLINK+ currently only support 2 GPU subset of 4 GPU case\n");
  }

  // Determine which user group is
  if ( ( devlist[ 0 ] == 0 && devlist[ 1 ] == 1 ) || ( devlist[ 0 ] == 1 && devlist[ 1 ] == 0 ) )
  {
    #ifndef NDEBUG
    printf("%s:: set user group 01\n", __func__ );
    #endif
  
    userGroupType = blinkplusUserGroupType::GROUP01;
  }
  else if ( ( devlist[ 0 ] == 1 && devlist[ 1 ] == 2 ) || ( devlist[ 0 ] == 2 && devlist[ 1 ] == 1 ) )
  {
    #ifndef NDEBUG
    printf("%s:: set user group 12\n", __func__ );
    #endif
  
    userGroupType = blinkplusUserGroupType::GROUP12;
  }
  else if ( ( devlist[ 0 ] == 0 && devlist[ 1 ] == 3 ) || ( devlist[ 0 ] == 3 && devlist[ 1 ] == 0 ) )
  {
    #ifndef NDEBUG
    printf("%s:: set user group 03\n", __func__ );
    #endif
  
    userGroupType = blinkplusUserGroupType::GROUP03;
  }
  else if ( ( devlist[ 0 ] == 2 && devlist[ 1 ] == 3 ) || ( devlist[ 0 ] == 3 && devlist[ 1 ] == 2 ) )
  {
    #ifndef NDEBUG
    printf("%s:: set user group 23\n", __func__ );
    #endif
  
    userGroupType = blinkplusUserGroupType::GROUP23;
  }
  else if ( ( devlist[ 0 ] == 0 && devlist[ 1 ] == 2 ) || ( devlist[ 0 ] == 2 && devlist[ 1 ] == 0 ) )
  {
    #ifndef NDEBUG
    printf("%s:: set user group 02\n", __func__ );
    #endif
  
    userGroupType = blinkplusUserGroupType::GROUP02;
  }
  else if ( ( devlist[ 0 ] == 1 && devlist[ 1 ] == 3 ) || ( devlist[ 0 ] == 3 && devlist[ 1 ] == 1 ) )
  {
    #ifndef NDEBUG
    printf("%s:: set user group 13\n", __func__ );
    #endif
  
    userGroupType = blinkplusUserGroupType::GROUP13;
  }
  else
  {
    throw std::runtime_error("User GPU group currently not supported\n");
  }

  *helper_cnt = configs.at( userGroupType ).size();

  return ncclSuccess;
}


// NCCL_API(ncclResult_t, blinkplusCommInitAll, ncclComm_t* comms, int ndev, const int* devlist );
ncclResult_t blinkplusCommInitAll( ncclComm_t* comms, int ndev, const int *devlist )
{
  // Currently, lots of support is hardcoded
  // if (ndev != 2 ) 
  // {
  //   //WARN("Invalid device count requested : %d, need 2", ndev);
  //   return ncclInvalidArgument;
  // }
  if ( ndev != 2 )
  {
    throw std::runtime_error("BLINK+ currently only support 2 GPU subset of 4 GPU case\n");
  }

  // Create blink+ helper group
  // return its corresbonding error code
  blinkplusHelperGroupsContainer.clear();
  blinkplusHelperGroupsContainer.reserve( configs.at( userGroupType ).size() );
  setenv( "NCCL_PROTO", "Simple", 1);
  setenv( "NCCL_ALGO", "Tree", 1 );

  for ( const auto& config_i : configs.at( userGroupType ) )
  {
    #ifndef NDEBUG
    printf("%s:: build blink+ helper group with graph %s\n", __func__, config_i.first.c_str() );
    #endif

    blinkplusHelperGroupsContainer.emplace_back( config_i.first.c_str(), config_i.second );
  }

  // Save user graph file for later use
  std::string userGraphFile;
  if ( std::getenv("NCCL_GRAPH_FILE") != nullptr )
  {
    userGraphFile = std::string( std::getenv("NCCL_GRAPH_FILE") );
  }

  // Create user helper group
  for ( int group_i = 0; group_i < blinkplusHelperGroupsContainer.size(); ++group_i )
  {
    #define helperGroupI( i ) blinkplusHelperGroupsContainer.at( i )
  
    // Allocate memory buffer & stream
    for ( int comm_j = 0; comm_j < helperGroupI( group_i ).num_comms; comm_j++ )
    {
      // Check if use user buffer or not. 
      // User will provide buffer at runtime
      bool use_user_buffer = false;
      for ( int user_comm_k = 0; user_comm_k < ndev; ++user_comm_k )
      {
        if ( devlist[ user_comm_k ] == helperGroupI( group_i ).devs[ comm_j ] )
        {
          use_user_buffer = true;
          break;
        }
      }

      // Use our internal buffer
      CUDACHECK(cudaSetDevice( helperGroupI( group_i ).devs[ comm_j ] ));
      if ( !use_user_buffer )
      {
        #ifndef NDEBUG
        printf("%s:: helper group %d, comm %d, dev %d use blink+ internal buffer\n", \
          __func__, group_i, comm_j, helperGroupI( group_i ).devs[ comm_j ] );
        #endif

        CUDACHECK(cudaMalloc( &(helperGroupI( group_i ).sendbuff.at( comm_j )), BLINKPLUS_BUFFER_SIZE_BYTES ));
        CUDACHECK(cudaMalloc( &(helperGroupI( group_i ).recvbuff.at( comm_j )), BLINKPLUS_BUFFER_SIZE_BYTES ));
        CUDACHECK(cudaMemset(   helperGroupI( group_i ).sendbuff.at( comm_j ), 0, BLINKPLUS_BUFFER_SIZE_BYTES ));
        CUDACHECK(cudaMemset(   helperGroupI( group_i ).recvbuff.at( comm_j ), 0, BLINKPLUS_BUFFER_SIZE_BYTES ));
      }
      // set nullptr for safety
      else
      {
        #ifndef NDEBUG
        printf("%s:: helper group %d, comm %d, dev %d use user buffer\n", \
          __func__, group_i, comm_j, helperGroupI( group_i ).devs[ comm_j ] );
        #endif

        helperGroupI( group_i ).sendbuff.at( comm_j ) = nullptr;
        helperGroupI( group_i ).recvbuff.at( comm_j ) = nullptr;
      }
      CUDACHECK(cudaStreamCreate( &(helperGroupI( group_i ).streams.at( comm_j )) ));
    } // end for each comm/dev inside group


    // Initialize communicator
    setenv( "NCCL_GRAPH_FILE", helperGroupI( group_i ).graph_filepath.c_str() , 1 );
    NCCLCHECK( ncclCommInitAll( helperGroupI( group_i ).comms.data(), \
                                helperGroupI( group_i ).num_comms, \
                                helperGroupI( group_i ).devs.data() ) );

    #undef helperGroupI
  } // end for each helper group

  // Unset NCCL graph file
  if ( !userGraphFile.empty() )
  {
    setenv( "NCCL_GRAPH_FILE", userGraphFile.c_str(), 1 );
  }

  return ncclSuccess;
} // end of blinkplusCommInitAll


// NCCL_API(ncclResult_t, blinkplusCommDestroy, ncclComm_t* comms );
ncclResult_t blinkplusCommDestroy( ncclComm_t* comms, int ndev, const int *devlist )
{
  for ( int group_i = 0; group_i < blinkplusHelperGroupsContainer.size(); ++group_i )
  {
    #define helperGroupI( i ) blinkplusHelperGroupsContainer.at( i )
  
    for ( int comm_j = 0; comm_j < helperGroupI( group_i ).num_comms; comm_j++ )
    {
      bool is_user_buffer = false;
      for ( int user_comm_k = 0; user_comm_k < ndev; ++user_comm_k )
      {
        if ( devlist[ user_comm_k ] == helperGroupI( group_i ).devs[ comm_j ] )
        {
          is_user_buffer = true;
          break;
        }
      }

      // destroy blink+ internal buffer
      if ( !is_user_buffer )
      {
        CUDACHECK(cudaSetDevice( helperGroupI( group_i ).devs.at( comm_j ) ));
        CUDACHECK(cudaFree( helperGroupI( group_i ).sendbuff.at( comm_j ) ));
        CUDACHECK(cudaFree( helperGroupI( group_i ).recvbuff.at( comm_j ) ));
      }

      // destroy nccl comm used by blink+
      ncclCommDestroy( helperGroupI( group_i ).comms.at( comm_j ) );
    } // end for each comm/dev inside group

    #undef helperGroupI
  } // end for each helper group

  return ncclSuccess;
} // end of blinkplusCommDestroy


//NCCL_API(ncclResult_t, blinkplusBroadcast, ncclComm_t* comms, int ndev, const int* devlist, \
  const void*** sendbuff, void*** recvbuff, int* count, ncclDataType_t datatype, int root );
ncclResult_t blinkplusBroadcast( ncclComm_t* comms, int ndev, const int *devlist, \
    const void*** sendbuff, void*** recvbuff, int* count, ncclDataType_t datatype, int root )
{
  // Start broadcast for each group
  for ( int group_i = 0; group_i < blinkplusHelperGroupsContainer.size(); ++group_i )
  {
    NCCLCHECK(ncclGroupStart());
    #define helperGroupI( i ) blinkplusHelperGroupsContainer.at( i )
  
    // Check if user user buffer or own buffer
    for ( int comm_j = 0; comm_j < helperGroupI( group_i ).num_comms; comm_j++ )
    {
      bool use_user_buffer = false;
      for ( int user_comm_k = 0; user_comm_k < ndev; ++user_comm_k )
      {
        if ( devlist[ user_comm_k ] == helperGroupI( group_i ).devs[ comm_j ] )
        {
          #ifndef NDEBUG
          printf("%s::run broadcast group %d comm %d with user buffer\n", __func__, group_i, comm_j );
          #endif

          use_user_buffer = true;
          NCCLCHECK(ncclBroadcast((const void*)sendbuff[ group_i ][ user_comm_k ], \
                                  (void*)recvbuff[ group_i ][ user_comm_k ], \
                                  count[ group_i ], datatype, root, \
                                  helperGroupI( group_i ).comms.at( comm_j ), \
                                  helperGroupI( group_i ).streams.at( comm_j )));
          break;
        }
      }

      if ( !use_user_buffer )
      {
        #ifndef NDEBUG
        printf("%s::run broadcast group %d comm %d with blink internal buffer\n", __func__, group_i, comm_j );
        #endif

        NCCLCHECK(ncclBroadcast((const void*)helperGroupI( group_i ).sendbuff.at( comm_j ), \
                                (void*)helperGroupI( group_i ).recvbuff.at( comm_j ), \
                                count[ group_i ], datatype, root, \
                                helperGroupI( group_i ).comms.at( comm_j ), \
                                helperGroupI( group_i ).streams.at( comm_j ) ));
      }
    } // end for each comm/dev inside group

    #undef helperGroupI

    NCCLCHECK(ncclGroupEnd());
  } // end for each helper group

  return ncclSuccess;
}


// NCCL_API(ncclResult_t, blinkplusAllReduce, ncclComm_t* comms, int ndev, const int* devlist, \
  const void*** sendbuff, void*** recvbuff, int* count, ncclDataType_t datatype, ncclRedOp_t op );
ncclResult_t  blinkplusAllReduce( ncclComm_t* comms, int ndev, const int *devlist, \
    const void*** sendbuff, void*** recvbuff, int* count, ncclDataType_t datatype, ncclRedOp_t op )
{
  // Start all reduce for each group
  for ( int group_i = 0; group_i < blinkplusHelperGroupsContainer.size(); ++group_i )
  {
    NCCLCHECK(ncclGroupStart());

    #define helperGroupI( i ) blinkplusHelperGroupsContainer.at( i )
  
    // Check if user user buffer or own buffer
    for ( int comm_j = 0; comm_j < helperGroupI( group_i ).num_comms; comm_j++ )
    {
      bool use_user_buffer = false;
      for ( int user_comm_k = 0; user_comm_k < ndev; ++user_comm_k )
      {
        if ( devlist[ user_comm_k ] == helperGroupI( group_i ).devs[ comm_j ] )
        {
          use_user_buffer = true;
          NCCLCHECK(ncclAllReduce((const void*)sendbuff[ group_i ][ user_comm_k ], \
                                  (void*)recvbuff[ group_i ][ user_comm_k ], \
                                  count[ group_i ], datatype, op, \
                                  helperGroupI( group_i ).comms.at( comm_j ), \
                                  helperGroupI( group_i ).streams.at( comm_j )));
          break;
        }
      }

      if ( !use_user_buffer )
      {
        NCCLCHECK(ncclAllReduce((const void*)helperGroupI( group_i ).sendbuff.at( comm_j ), \
                                (void*)helperGroupI( group_i ).recvbuff.at( comm_j ), \
                                count[ group_i ], datatype, op, \
                                helperGroupI( group_i ).comms.at( comm_j ), \
                                helperGroupI( group_i ).streams.at( comm_j ) ));
      }
    } // end for each comm/dev inside group

    #undef helperGroupI

    NCCLCHECK(ncclGroupEnd());
  } // end for each helper group

  return ncclSuccess;
}


// NCCL_API(ncclResult_t, blinkplusStreamSynchronize, ncclComm_t* comms );
ncclResult_t blinkplusStreamSynchronize( ncclComm_t* comms )
{
  for ( int group_i = 0; group_i < blinkplusHelperGroupsContainer.size(); ++group_i )
  {
    #define helperGroupI( i ) blinkplusHelperGroupsContainer.at( i )
  
    // Synchronize for every communicator
    for ( int comm_j = 0; comm_j < helperGroupI( group_i ).num_comms; comm_j++ )
    {
      #ifndef NDEBUG
      printf("%s:: group %d, comm %d call stream sync on dev %d\n", __func__, group_i, comm_j, helperGroupI( group_i ).devs.at( comm_j ) );
      #endif

      CUDACHECK(cudaSetDevice( helperGroupI( group_i ).devs.at( comm_j ) ));
      CUDACHECK(cudaStreamSynchronize( helperGroupI( group_i ).streams.at( comm_j ) ));
    } // end for each comm/dev inside group

    #undef helperGroupI
  } // end for each helper group

  return ncclSuccess;
}

#undef CUDACHECK
#undef NCCLCHECK