#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include <stdlib.h>
#include<chrono>

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

int main(int argc, char *argv[]) {

    // setenv( "NCCL_PROTO", "Simple", 1);
    // setenv( "NCCL_DEBUG", "Info", 1);
    // setenv( "NCCL_DEBUG_SUBSYS", "ALL", 1);
    // setenv( "NCCL_ALGO", "Ring", 1 );

    // managing 4 devices
    constexpr int nDev = 2;
    int size = 512 * 1024 * 1024;
    int devs[nDev] = {0, 1};
    ncclComm_t comms[nDev];

    // allocating and initializing device buffers
    int **sendbuff = (int **) malloc(nDev * sizeof(int *));
    int **recvbuff = (int **) malloc(nDev * sizeof(int *));
    cudaStream_t *s = (cudaStream_t *) malloc(sizeof(cudaStream_t) * nDev);
    
    int *h_sendbuff = (int*)malloc(size * sizeof(int));
    for ( int i = 0; i < size; ++i )
    {
      h_sendbuff[ i ] = i;
    }

    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(int)));
        CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(int)));
        //CUDACHECK(cudaMemset(sendbuff[i], 100, size * sizeof(int)));
        CUDACHECK(cudaMemcpy(sendbuff[i], h_sendbuff, size * sizeof(int), cudaMemcpyHostToDevice ));
        CUDACHECK(cudaMemset(recvbuff[i], -1, size * sizeof(int)));
        CUDACHECK(cudaStreamCreate(s + i));
    }

    // initializing NCCL
    NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

    // Start timing, this is only a rouph timing
    int start = getTime();

    // calling NCCL communication API. Group API is required when
    // using multiple devices per thread
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nDev; ++i) {
        NCCLCHECK(ncclBroadcast((const void *) sendbuff[i],
                                (void *) recvbuff[i], size, ncclInt, 0,
                                comms[i], s[i]));
    }
    NCCLCHECK(ncclGroupEnd());

    // synchronizing on CUDA streams to wait for completion of NCCL operation
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(s[i]));
    }

    // End timing
    // This is only rouphy timing
    int elasped = getTime() - start;
    printf("%s:: Elapsed Time: %.d \n", __func__, elasped);

    // Copy data back to check data
    int *h_recvbuff = (int*)malloc(size * sizeof(int));
    for ( int i = 0; i < nDev; ++i ) 
    {
      CUDACHECK(cudaSetDevice(i));
      CUDACHECK(cudaMemcpy( h_recvbuff, recvbuff[i], size * sizeof(int), cudaMemcpyDeviceToHost ));

      for ( int j = 0; j < size; ++j )
      {
        if ( h_recvbuff[j] != h_sendbuff[j] )
        {
          printf("comm %d expected %d but have %d\n", i, h_sendbuff[j], h_recvbuff[j]);
        }
      }
    }

    // free device buffers
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaFree(sendbuff[i]));
        CUDACHECK(cudaFree(recvbuff[i]));
    }

    free(h_sendbuff);

    // finalizing NCCL
    for (int i = 0; i < nDev; ++i) {
        ncclCommDestroy(comms[i]);
    }

    printf("Success \n");
    return 0;
}