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

void runRoll(int nDev, int **sendbuff, int **recvbuff, int size, ncclComm_t* comms, cudaStream_t* s) {
    // calling NCCL communication API. Group API is required when
    // using multiple devices per thread
    printf("Roll Start\n");
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
}


int main(int argc, char *argv[]) {

    // setenv( "NCCL_PROTO", "Simple", 1);
    // setenv( "NCCL_DEBUG", "Info", 1);
    // setenv( "NCCL_DEBUG_SUBSYS", "ALL", 1);
    // setenv( "NCCL_ALGO", "Ring", 1 );

    // managing 4 devices
    constexpr int nDev = 2;
    int size = atoi(argv[4]) * 1024 * 1024;
    int warm_up_iters = 5;
    int iters = atoi(argv[3]);
    int devs[nDev] = {atoi(argv[1]), atoi(argv[2])};
    ncclComm_t comms[nDev];

    printf("=========%s:: Initial data of size %d MB=========\n", __func__, int(atoi(argv[4]) * 4));

    // allocating and initializing device buffers
    int **sendbuff = (int **) malloc(nDev * sizeof(int *));
    int **recvbuff = (int **) malloc(nDev * sizeof(int *));
    cudaStream_t *s = (cudaStream_t *) malloc(sizeof(cudaStream_t) * nDev);
    
    int *h_sendbuff = (int*)malloc(size * sizeof(int));
    int right_ans = 0;
    for (int i = 0; i < nDev; ++i) {
      for ( int j = 0; j < size; ++j )
        {
          h_sendbuff[ j ] = 21 + i * 5;   // 21  + 26 = 47
        }
        right_ans += h_sendbuff[0];
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(int)));
        CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(int)));
        //CUDACHECK(cudaMemset(sendbuff[i], 100, size * sizeof(int)));
        CUDACHECK(cudaMemcpy(sendbuff[i], h_sendbuff, size * sizeof(int), cudaMemcpyHostToDevice ));
        CUDACHECK(cudaMemset(recvbuff[i], -1, size * sizeof(int)));
        CUDACHECK(cudaStreamCreate(s + i));
    }
    printf("all reduce correct ans: %d \n", right_ans);

    // initializing NCCL
    NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

    printf("Start WarmUp, Iters: %d \n", warm_up_iters);
    for (int iter = 0; iter < warm_up_iters; iter++) {
      NCCLCHECK(ncclGroupStart());
      for (int i = 0; i < nDev; ++i) {
          NCCLCHECK(ncclAllReduce((const void *) sendbuff[i],
                                  (void *) recvbuff[i], size, ncclInt, ncclSum,
                                  comms[i], s[i]));
      }
      NCCLCHECK(ncclGroupEnd());

      // synchronizing on CUDA streams to wait for completion of NCCL operation
      for (int i = 0; i < nDev; ++i) {
          CUDACHECK(cudaSetDevice(i));
          CUDACHECK(cudaStreamSynchronize(s[i]));
      }
    }
    printf("End WarmUp\n");

    // Start timing, Run multiple times
    //int start = getTime();
    printf("=====Start Timing, Iters: %d======\n", iters);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iters; iter++) {
      NCCLCHECK(ncclGroupStart());
      for (int i = 0; i < nDev; ++i) {
          NCCLCHECK(ncclAllReduce((const void *) sendbuff[i],
                                  (void *) recvbuff[i], size, ncclInt, ncclSum,
                                  comms[i], s[i]));
      }
      NCCLCHECK(ncclGroupEnd());

      // synchronizing on CUDA streams to wait for completion of NCCL operation
      for (int i = 0; i < nDev; ++i) {
          CUDACHECK(cudaSetDevice(i));
          CUDACHECK(cudaStreamSynchronize(s[i]));
      }
    }

    printf("=====End Timing======\n");
    auto delta = std::chrono::high_resolution_clock::now() - start;
    double deltaSec = std::chrono::duration_cast<std::chrono::duration<double>>(delta).count();
    deltaSec = deltaSec / iters;
    double timeUsec = deltaSec*1.0E6;
    double bw = size * sizeof(int) / 1.0E9 / deltaSec;
    printf("%s:: Average of %d Iters, data: %d MB,  Elapsed Time: %7.5f (us), BandWidth: %7.5f (GB/s)\n", \
                __func__, iters, int(atoi(argv[4]) * 4), timeUsec,  bw);

    // Copy data back to check data
    int *h_recvbuff = (int*)malloc(size * sizeof(int));
    for ( int i = 0; i < nDev; ++i ) 
    {
      CUDACHECK(cudaSetDevice(i));
      CUDACHECK(cudaMemcpy( h_recvbuff, recvbuff[i], size * sizeof(int), cudaMemcpyDeviceToHost ));

      for ( int j = 0; j < size; ++j )
      { 
        //printf("comm %d have %d\n", i, h_recvbuff[j]);
        if ( h_recvbuff[j] != right_ans)
        {
          printf("comm %d expected %d but have %d\n", i, right_ans, h_recvbuff[j]);
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