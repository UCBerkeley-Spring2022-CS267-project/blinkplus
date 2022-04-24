## Compile

```shell
mkdir build
cd build

# build on bridge 2
cmake ..

# build on rise
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.5/bin/nvcc -DCMAKE_BUILD_TYPE=Release

make -j 10

```