cmake_minimum_required(VERSION 3.0 )

set(CMAKE_CUDA_ARCHITECTURES 72)
set(CMAKE_BUILD_TYPE Release)

# project name
project( blinkplus LANGUAGES CXX CUDA )
message(STATUS "CMAKE_CUDA_COMPILER ${CMAKE_CUDA_COMPILER}")

# find nccl
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/)

# Require NCCL
find_package(nccl REQUIRED)
if ( NCCL_FOUND )
    message(STATUS "NCCL Found ${NCCL_INCLUDE_DIRS} ${NCCL_LIBRARIES}")
else()
    message(STATUS "NCCL Not Found")
endif()

# header
include_directories( ${NCCL_INCLUDE_DIRS} )
include_directories( include )

# Build library
add_library(${PROJECT_NAME} STATIC src/blinkplus.cu )
target_link_libraries(${PROJECT_NAME} PRIVATE ${nccl} )
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_11 )

# Test bandwidth
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/test_bandwidth )

# NCCL Broadcast Bandwidth
add_executable( time_nccl_broadcast_2gpu.out test_bandwidth/time_nccl_broadcast_2gpu.cu )
target_compile_features( time_nccl_broadcast_2gpu.out PRIVATE cxx_std_11)
target_link_libraries( time_nccl_broadcast_2gpu.out PRIVATE nccl )

add_executable( time_nccl_broadcast_3gpu.out test_bandwidth/time_nccl_broadcast_3gpu.cu )
target_compile_features( time_nccl_broadcast_3gpu.out PRIVATE cxx_std_11)
target_link_libraries( time_nccl_broadcast_3gpu.out PRIVATE nccl )

# NCCL Allreduce Bandwidth
add_executable( time_nccl_allreduce_2gpu.out test_bandwidth/time_nccl_allreduce_2gpu.cu )
target_compile_features( time_nccl_allreduce_2gpu.out PRIVATE cxx_std_11)
target_link_libraries( time_nccl_allreduce_2gpu.out PRIVATE nccl )

# Mock BLINK+ Bandwidth
add_executable( time_nccl_broadcast_user01_2group.out test_bandwidth/time_nccl_broadcast_user01_2group.cu )
target_compile_features( time_nccl_broadcast_user01_2group.out PRIVATE cxx_std_11)
target_link_libraries( time_nccl_broadcast_user01_2group.out PRIVATE nccl )

add_executable( time_nccl_broadcast_user01_3group.out test_bandwidth/time_nccl_broadcast_user01_3group.cu )
target_compile_features( time_nccl_broadcast_user01_3group.out PRIVATE cxx_std_11)
target_link_libraries( time_nccl_broadcast_user01_3group.out PRIVATE nccl )

add_executable( time_nccl_broadcast_user01_3group_complex.out test_bandwidth/time_nccl_broadcast_user01_3group_complex.cu )
target_compile_features( time_nccl_broadcast_user01_3group_complex.out PRIVATE cxx_std_11)
target_link_libraries( time_nccl_broadcast_user01_3group_complex.out PRIVATE nccl )

add_executable( time_nccl_allreduce_user01_3group.out test_bandwidth/time_nccl_allreduce_user01_3group.cu )
target_compile_features( time_nccl_allreduce_user01_3group.out PRIVATE cxx_std_11)
target_link_libraries( time_nccl_allreduce_user01_3group.out PRIVATE nccl )

add_executable( time_nccl_broadcast_user01_4group.out test_bandwidth/time_nccl_broadcast_user01_4group.cu )
target_compile_features( time_nccl_broadcast_user01_4group.out PRIVATE cxx_std_11)
target_link_libraries( time_nccl_broadcast_user01_4group.out PRIVATE nccl )

add_executable( time_nccl_broadcast_user02_2group.out test_bandwidth/time_nccl_broadcast_user02_2group.cu )
target_compile_features( time_nccl_broadcast_user02_2group.out PRIVATE cxx_std_11)
target_link_libraries( time_nccl_broadcast_user02_2group.out PRIVATE nccl )

add_executable( time_nccl_broadcast_user02_3group.out test_bandwidth/time_nccl_broadcast_user02_3group.cu )
target_compile_features( time_nccl_broadcast_user02_3group.out PRIVATE cxx_std_11)
target_link_libraries( time_nccl_broadcast_user02_3group.out PRIVATE nccl )

add_executable( time_nccl_broadcast_user02_4group.out test_bandwidth/time_nccl_broadcast_user02_4group.cu )
target_compile_features( time_nccl_broadcast_user02_4group.out PRIVATE cxx_std_11)
target_link_libraries( time_nccl_broadcast_user02_4group.out PRIVATE nccl )

add_executable( time_nccl_broadcast_user03_3group.out test_bandwidth/time_nccl_broadcast_user03_3group.cu )
target_compile_features( time_nccl_broadcast_user03_3group.out PRIVATE cxx_std_11)
target_link_libraries( time_nccl_broadcast_user03_3group.out PRIVATE nccl )

add_executable( time_nccl_allreduce_user03_3group.out test_bandwidth/time_nccl_allreduce_user03_3group.cu )
target_compile_features( time_nccl_allreduce_user03_3group.out PRIVATE cxx_std_11)
target_link_libraries( time_nccl_allreduce_user03_3group.out PRIVATE nccl )

add_executable( time_nccl_broadcast_user03_4group.out test_bandwidth/time_nccl_broadcast_user03_4group.cu )
target_compile_features( time_nccl_broadcast_user03_4group.out PRIVATE cxx_std_11)
target_link_libraries( time_nccl_broadcast_user03_4group.out PRIVATE nccl )

# BLINK+ Bandwidth
add_executable( time_blinkplus_broadcast.out test_bandwidth/time_blinkplus_broadcast.cu )
target_compile_features( time_blinkplus_broadcast.out PRIVATE cxx_std_11)
target_link_libraries( time_blinkplus_broadcast.out PRIVATE ${PROJECT_NAME} nccl )

add_executable( time_blinkplus_allreduce.out test_bandwidth/time_blinkplus_allreduce.cu )
target_compile_features( time_blinkplus_allreduce.out PRIVATE cxx_std_11)
target_link_libraries( time_blinkplus_allreduce.out PRIVATE ${PROJECT_NAME} nccl )
