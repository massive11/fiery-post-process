cmake_minimum_required(VERSION 3.16)
 
project(fiery_post_process)
find_package(PythonInterp REQUIRED)
set(APP_NAME fiery_post_process)

set(CMAKE_CXX_STANDARD 14)

set(Torch_DIR /home/yu/Git_repos/CLionProjects/libtorch/share/cmake/Torch)

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.1/bin/nvcc)

file(GLOB SOURCE_FILES /media/data/Git_repos/fiery_post_process/*.c /media/data/Git_repos/fiery_post_process/*.cpp /media/data/Git_repos/fiery_post_process/*.cxx /media/data/Git_repos/fiery_post_process/*.cc)
 
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

 
#set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/home/yu/Git_repos/CLionProjects/pytorch-1.12.0")

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/..
)

find_package(Torch REQUIRED)
#include_directories("/home/yu/Git_repos/CLionProjects/pytorch-1.12.0/include")
#include_directories("/home/yu/Git_repos/CLionProjects/libtorch/include/torch/csrc/api/include")

target_link_libraries(${APP_NAME} ${TORCH_LIBRARIES})