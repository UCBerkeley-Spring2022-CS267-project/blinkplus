cmake_minimum_required(VERSION 3.0 )

# project name
project( nccl_example LANGUAGES CXX CUDA )
message(STATUS "CMAKE_CUDA_COMPILER ${CMAKE_CUDA_COMPILER}")

# find nccl
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/)

# nccl
find_package(nccl REQUIRED)
if ( NCCL_FOUND )
    message(STATUS "!!!!NCCL Found ${NCCL_INCLUDE_DIRS} ${NCCL_LIBRARIES}!!!")
else()
    message(STATUS "!!!!NCCL Not Found!!")
endif()

# header
include_directories( ${NCCL_INCLUDE_DIRS} )

# exec
add_executable( mock_blinkplus_2group01 mock_blinkplus_2group01.cu  )
add_executable( mock_blinkplus_group01_group02 mock_blinkplus_group01_group02.cu  )
add_executable( mock_blinkplus_group01_group02_group03 mock_blinkplus_group01_group02_group03.cu  )
add_executable( test_launchkernel test_launchkernel.cu  )

#add_executable( demo_group01 demo_group01.cu  )
#add_executable( demo demo.cu  )

# I'm using C++11 in the test
target_compile_features( mock_blinkplus_2group01 PRIVATE cxx_std_11)
target_compile_features( mock_blinkplus_group01_group02 PRIVATE cxx_std_11)
target_compile_features( mock_blinkplus_group01_group02_group03 PRIVATE cxx_std_11)
target_compile_features( test_launchkernel PRIVATE cxx_std_11)

#target_compile_features( demo_group01 PRIVATE cxx_std_11)
#target_compile_features( demo PRIVATE cxx_std_11)

# link nccl
target_link_libraries( mock_blinkplus_2group01 PRIVATE ${NCCL_LIBRARIES} )
target_link_libraries( mock_blinkplus_group01_group02 PRIVATE ${NCCL_LIBRARIES} )
target_link_libraries( mock_blinkplus_group01_group02_group03 PRIVATE ${NCCL_LIBRARIES} )
target_link_libraries( test_launchkernel PRIVATE ${NCCL_LIBRARIES} )

#target_link_libraries( demo_group01 PRIVATE ${NCCL_LIBRARIES} )
#target_link_libraries( demo PRIVATE ${NCCL_LIBRARIES} )
