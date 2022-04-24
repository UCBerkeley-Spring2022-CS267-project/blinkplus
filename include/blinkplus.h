#pragma once

#include "nccl.h"

ncclResult_t blinkplusCommInitAll( ncclComm_t* comms, int ndev, const int *devlist, int numBlinkplusHelperGroup );

ncclResult_t  blinkplusCommDestroy( ncclComm_t* comms );

ncclResult_t  blinkplusBroadcast( ncclComm_t* comms, int ndev, const int *devlist, \
    const void*** sendbuff, void*** recvbuff, size_t* count, ncclDataType_t datatype, int root );

ncclResult_t  blinkplusAllReduce( ncclComm_t* comms, int ndev, const int *devlist, \
    const void*** sendbuff, void*** recvbuff, size_t* count, ncclDataType_t datatype, ncclRedOp_t op );

ncclResult_t blinkplusStreamSynchronize( ncclComm_t* comms );

