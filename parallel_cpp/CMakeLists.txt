cmake_minimum_required(VERSION 3.18)
project(sub_diffusion_fit_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # for VS Code IntelliSense

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# GTX 1650 Ti is Turing (compute capability 7.5)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(ZLIB REQUIRED)

# Host-side I/O is shared with the serial version
set(SERIAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../serial_cpp)

add_executable(fit_sub_cuda
    main.cu
    ${SERIAL_DIR}/read_mri.cpp
    ${SERIAL_DIR}/save_map.cpp
)

target_include_directories(fit_sub_cuda PRIVATE ${SERIAL_DIR})
target_link_libraries(fit_sub_cuda PRIVATE CUDA::cudart ZLIB::ZLIB)
