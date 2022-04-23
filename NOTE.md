## Common use command

* request 4 GPU interactive on bridge 2
```shell
salloc -N 1 -p GPU-shared --gres=gpu:4 -q interactive
```

* request node como on rise
```shell
srun --nodelist=como -t 60:00 --pty bash
```

* check nvlink connection
```shell
nvidia-smi topo -m
```

## Settings for nccl

```shell
# create config file
touch ~/.nccl.conf

# use level nccl runtime configuration
# nccl will read this file before run
vim ~/.nccl.conf
```

content of the config file
```conf
# DUMP topology
NCCL_TOPO_DUMP_FILE=`pwd`/topo.xml
# use simple proto instead of LL (low latency)
NCCL_PROTO=Simple
# show debug info
NCCL_DEBUG=Info
# debug subsystem
NCCL_DEBUG_SUBSYS=ALL
# choose algo
NCCL_ALGO=Ring
# debug file
NCCL_DEBUG_FILE=`pwd`/debugfile.%h.%p
# save graph file
NCCL_GRAPH_DUMP_FILE=`pwd`/graph.xml
```

## profiling
```shell
# save profile result
/usr/local/cuda-11.5/bin/nvprof -o output.nvvp ./examples/build/mock_blinkplus_group01_group02

# visual result
/usr/local/cuda-11.5/bin/nvvp output.nvvp
```

## Export graph path
```shell
export NCCL_GRAPH_FILE_CHAIN_01=`pwd`/graphs/chain01.xml
export NCCL_GRAPH_FILE_CHAIN_02=`pwd`/graphs/chain02.xml

export NCCL_GRAPH_FILE_CHAIN_021=`pwd`/graphs/chain021.xml
export NCCL_GRAPH_FILE_CHAIN_031=`pwd`/graphs/chain031.xml
export NCCL_GRAPH_FILE_CHAIN_0321=`pwd`/graphs/chain0321.xml
export NCCL_GRAPH_FILE_CHAIN_0123=`pwd`/graphs/chain0123.xml

echo $NCCL_GRAPH_FILE_CHAIN_01
echo $NCCL_GRAPH_FILE_CHAIN_021
echo $NCCL_GRAPH_FILE_CHAIN_031
echo $NCCL_GRAPH_FILE_CHAIN_0321
echo $NCCL_GRAPH_FILE_CHAIN_0123
```