#pragma once

#include "nccl.h"

// Currently only support creating blink+ group for one user group. 
// Currently do not support multiple user group

// Get number of helper group given user defined comms
// Now only support 2 gpu subset of 4 gpu group (e.g. gpu01, gpu12)
ncclResult_t blinkplusGetHelperCnt( ncclComm_t* comms, int ndev, const int *devlist, int* helper_cnt );

// Create blink+ helper groups
ncclResult_t blinkplusCommInitAll( ncclComm_t* comms, int ndev, const int *devlist );

// Destroy blink+ helper group
ncclResult_t  blinkplusCommDestroy( ncclComm_t* comms, int ndev, const int *devlist );

// BLINK+ helper group run broadcast
ncclResult_t  blinkplusBroadcast( ncclComm_t* comms, int ndev, const int *devlist, \
    const void*** sendbuff, void*** recvbuff, int* count, ncclDataType_t datatype, int root );

// BLINK+ helper group run allreduce
ncclResult_t  blinkplusAllReduce( ncclComm_t* comms, int ndev, const int *devlist, \
    const void*** sendbuff, void*** recvbuff, int* count, ncclDataType_t datatype, ncclRedOp_t op );

// BLINK+ internal stream synchronize
ncclResult_t blinkplusStreamSynchronize( ncclComm_t* comms );

