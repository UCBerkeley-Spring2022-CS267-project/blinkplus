cmake_minimum_required(VERSION 3.0 )

set(CMAKE_CUDA_ARCHITECTURES 72)
# project name
project( blinkplus LANGUAGES CXX CUDA )
message(STATUS "CMAKE_CUDA_COMPILER ${CMAKE_CUDA_COMPILER}")

# find nccl
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/)

# nccl
# find_package(nccl REQUIRED)
# if ( NCCL_FOUND )
#     message(STATUS "NCCL Found ${NCCL_INCLUDE_DIRS} ${NCCL_LIBRARIES}")
# else()
#     message(STATUS "NCCL Not Found")
# endif()

# header
# include_directories( ${NCCL_INCLUDE_DIRS} )
include_directories( include )

# exec
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/blinkplus_examples )

add_executable( run_blinkplus blinkplus_examples/run_blinkplus.cu src/blinkplus.cu )
add_executable( run_blinkplus_broadcast blinkplus_examples/run_blinkplus_broadcast.cu src/blinkplus.cu )
add_executable( run_blinkplus_allreduce blinkplus_examples/run_blinkplus_allreduce.cu src/blinkplus.cu )
add_executable( run_blinkplus_broadcast_12 blinkplus_examples/run_blinkplus_broadcast_12.cu src/blinkplus.cu )


add_executable( run_nccl blinkplus_examples/run_nccl.cu src/blinkplus.cu )
add_executable( run_nccl_allreduce blinkplus_examples/run_nccl_allreduce.cu src/blinkplus.cu)




# I'm using C++11 in the test
target_compile_features( run_blinkplus PRIVATE cxx_std_11)
target_compile_features( run_blinkplus_broadcast PRIVATE cxx_std_11)
target_compile_features( run_blinkplus_broadcast_12 PRIVATE cxx_std_11)
target_compile_features( run_blinkplus_allreduce PRIVATE cxx_std_11)

target_compile_features( run_nccl PRIVATE cxx_std_11)
target_compile_features( run_nccl_allreduce PRIVATE cxx_std_11)


# link nccl
target_link_libraries( run_blinkplus PRIVATE nccl )
target_link_libraries( run_blinkplus_broadcast PRIVATE nccl )
target_link_libraries( run_blinkplus_allreduce PRIVATE nccl)
target_link_libraries( run_blinkplus_broadcast_12 PRIVATE nccl)


target_link_libraries( run_nccl PRIVATE nccl )
target_link_libraries( run_nccl_allreduce PRIVATE nccl )



# target_link_libraries( run_blinkplus PRIVATE ${NCCL_LIBRARIES} )
# target_link_libraries( run_blinkplus_broadcast PRIVATE ${NCCL_LIBRARIES} )
# target_link_libraries( run_nccl PRIVATE ${NCCL_LIBRARIES} )
