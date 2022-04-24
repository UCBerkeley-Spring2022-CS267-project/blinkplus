# BLINK+



## Install

* Build blink+ package and intall package
```shell
mkdir build
cd build
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.5/bin/nvcc -DCMAKE_BUILD_TYPE=Release
make -j 10
sudo make install
```

* Build blink+ example
```shell
cd blinkplus_example
cd build
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.5/bin/nvcc -DCMAKE_BUILD_TYPE=Release
make -j 10
```

* Export graph path

```shell
export BLINKPLUS_GRAPH_FILE_CHAIN_01=`pwd`/graphs/chain01.xml
export BLINKPLUS_GRAPH_FILE_CHAIN_02=`pwd`/graphs/chain02.xml
export BLINKPLUS_GRAPH_FILE_CHAIN_03=`pwd`/graphs/chain03.xml
export BLINKPLUS_GRAPH_FILE_CHAIN_23=`pwd`/graphs/chain23.xml
export BLINKPLUS_GRAPH_FILE_CHAIN_021=`pwd`/graphs/chain021.xml
export BLINKPLUS_GRAPH_FILE_CHAIN_031=`pwd`/graphs/chain031.xml
export BLINKPLUS_GRAPH_FILE_CHAIN_0321=`pwd`/graphs/chain0321.xml
export BLINKPLUS_GRAPH_FILE_CHAIN_0123=`pwd`/graphs/chain0123.xml
```
