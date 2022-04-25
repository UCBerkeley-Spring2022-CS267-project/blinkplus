# BLINK+



## Install

* Build nccl package v2.7.8

We have prepared a NCCL package v2.7.8 with some bug fix [link](https://github.com/UCBerkeley-Spring2022-CS267-project/blinkplus-nccl-base)

```shell
git clone https://github.com/UCBerkeley-Spring2022-CS267-project/blinkplus-nccl-base.git
cd blinkplus-nccl-base
```

and then follow the INSTALL.md file here [link](https://github.com/UCBerkeley-Spring2022-CS267-project/blinkplus-nccl-base/blob/blinkplus_base_v2.7.8/INSTALL.md) to install NCCL

* Build blink+ example
```shell
mkdir build
cd build
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.5/bin/nvcc -DCMAKE_BUILD_TYPE=Release
make -j 10
sudo make install
```

The example is currently placed under `blinkplus_examples`


* Export graph path

```shell
export BLINKPLUS_GRAPH_FILE_CHAIN_01=`pwd`/graphs/chain01.xml
export BLINKPLUS_GRAPH_FILE_CHAIN_02=`pwd`/graphs/chain02.xml
export BLINKPLUS_GRAPH_FILE_CHAIN_03=`pwd`/graphs/chain03.xml
export BLINKPLUS_GRAPH_FILE_CHAIN_23=`pwd`/graphs/chain23.xml
export BLINKPLUS_GRAPH_FILE_CHAIN_021=`pwd`/graphs/chain021.xml
export BLINKPLUS_GRAPH_FILE_CHAIN_031=`pwd`/graphs/chain031.xml
export BLINKPLUS_GRAPH_FILE_CHAIN_102=`pwd`/graphs/chain102.xml
export BLINKPLUS_GRAPH_FILE_CHAIN_123=`pwd`/graphs/chain123.xml
export BLINKPLUS_GRAPH_FILE_CHAIN_0321=`pwd`/graphs/chain0321.xml
export BLINKPLUS_GRAPH_FILE_CHAIN_0123=`pwd`/graphs/chain0123.xml
```
