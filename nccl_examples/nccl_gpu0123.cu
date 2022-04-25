#include <stdio.h>
#include <array>
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

int main(int argc, char* argv[])
{
  // set enviroment variable before run
  // this is program level setting and thus do not pollute global 
  setenv( "NCCL_PROTO", "Simple", 1);
  setenv( "NCCL_DEBUG", "Info", 1);
  setenv( "NCCL_DEBUG_SUBSYS", "ALL", 1);
  setenv( "NCCL_ALGO", "Tree", 1 ); // Tree : AllReduceTree+BroadcastRing | Ring : AllReduceRing+BroadcastRing

  // managing 4 devices
  int size;
  if ( argc >= 2 )
    size = atoi( argv[ 1 ] );
  else
    size = 4*1024*1024;

  const std::vector<int> devs = { 0,1,2,3 };
  int nDevs = 4;
  ncclComm_t comms[nDevs];
  printf("Using %d GPU for test on size %d\n", nDevs, size ); fflush(stdout);

  printf("Allocate send & recv buffer\n"); fflush(stdout);
  // allocating and initializing device buffers
  float** sendbuff = (float**)malloc(nDevs * sizeof(float*));
  float** recvbuff = (float**)malloc(nDevs * sizeof(float*));
  // stream is within current GPU.
  // We're using 3 GPU and thus should have 3 stream.
  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*(nDevs));

  // Allocate memory
  for ( int i = 0; i < nDevs; ++i ) 
  {
    CUDACHECK(cudaSetDevice( devs[i] ));
    CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float))); // &(sendbuff[i])
    CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
    CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(s+i));
  }

  // Set enviroment variable to search
  if ( std::getenv("NCCL_GRAPH_FILE_CHAIN_0123") == nullptr )
  {
     throw std::runtime_error("NCCL_GRAPH_FILE_CHAIN_0123 not set\b");
  }

  setenv( "NCCL_GRAPH_FILE", std::getenv("NCCL_GRAPH_FILE_CHAIN_0123") , 1 );

  //initializing NCCL
  printf("Initial comm\n"); fflush(stdout);
  NCCLCHECK(ncclCommInitAll(comms, nDevs, devs.data()));

   //calling NCCL communication API. Group API is required when using
   //multiple devices per thread
  printf("Run call reduce\n"); fflush(stdout);
  NCCLCHECK(ncclGroupStart());
  for ( int i = 0; i < nDevs; ++i ) 
  {
    // allreduce
    NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum, comms[i], s[i]));
  }
  NCCLCHECK(ncclGroupEnd());

  //synchronizing on CUDA streams to wait for completion of NCCL operation
  printf("stream synchronize\n"); fflush(stdout);
  for ( int i = 0; i < nDevs; ++i ) 
  {
    CUDACHECK(cudaSetDevice(devs[i]));
    CUDACHECK(cudaStreamSynchronize(s[i]));
  }

  printf("Run broadcast\n"); fflush(stdout);
  NCCLCHECK(ncclGroupStart());
  for ( int i = 0; i < nDevs; ++i ) 
  {
    // broadcast
    NCCLCHECK(ncclBroadcast((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, 0, comms[i], s[i]));
  }
  NCCLCHECK(ncclGroupEnd());

  //synchronizing on CUDA streams to wait for completion of NCCL operation
  printf("stream synchronize\n"); fflush(stdout);
  for ( int i = 0; i < nDevs; ++i ) 
  {
    CUDACHECK(cudaSetDevice(devs[i]));
    CUDACHECK(cudaStreamSynchronize(s[i]));
  }

  //free device buffers
  printf("free used buffer\n"); fflush(stdout);
  for ( int i = 0; i < nDevs; ++i )
  {
    CUDACHECK(cudaSetDevice(devs[i]));
    CUDACHECK(cudaFree(sendbuff[i]));
    CUDACHECK(cudaFree(recvbuff[i]));
  }

  //finalizing NCCL
  printf("free comm buffer\n"); fflush(stdout);
  for ( int i = 0; i < nDevs; ++i )
  {
      printf("free %d/%d\n", i, nDevs ); fflush(stdout);
      ncclCommDestroy(comms[i]);
  }

  printf("Success \n");
  return 0;
}