cmake_minimum_required(VERSION 3.19)
project(cuda-example CUDA)
set(CMAKE_CUDA_STANDARD 14)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "-G -g")
endif()

add_executable(cuda-example main.cu)
set_target_properties(
        cuda-example
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
