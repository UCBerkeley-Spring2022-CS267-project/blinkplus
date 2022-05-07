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
export BLINKPLUS_GRAPH_FILE_CHAIN_021=`pwd`/graphs/user01/chain021.xml
export BLINKPLUS_GRAPH_FILE_CHAIN_031=`pwd`/graphs/user01/chain031.xml
export BLINKPLUS_GRAPH_FILE_CHAIN_0321=`pwd`/graphs/user01/chain0321.xml

export BLINKPLUS_GRAPH_FILE_CHAIN_102=`pwd`/graphs/user12/chain102.xml
export BLINKPLUS_GRAPH_FILE_CHAIN_132=`pwd`/graphs/user12/chain132.xml

export BLINKPLUS_GRAPH_FILE_CHAIN_013=`pwd`/graphs/user03/chain013.xml
export BLINKPLUS_GRAPH_FILE_CHAIN_023=`pwd`/graphs/user03/chain023.xml

export BLINKPLUS_GRAPH_FILE_CHAIN_203=`pwd`/graphs/user23/chain203.xml
export BLINKPLUS_GRAPH_FILE_CHAIN_213=`pwd`/graphs/user23/chain213.xml
export BLINKPLUS_GRAPH_FILE_CHAIN_2103=`pwd`/graphs/user23/chain2103.xml

export BLINKPLUS_GRAPH_FILE_CHAIN_032=`pwd`/graphs/user02/chain032.xml
export BLINKPLUS_GRAPH_FILE_CHAIN_012=`pwd`/graphs/user02/chain012.xml

export BLINKPLUS_GRAPH_FILE_CHAIN_103=`pwd`/graphs/user13/chain103.xml
export BLINKPLUS_GRAPH_FILE_CHAIN_123=`pwd`/graphs/user13/chain123.xml
```
