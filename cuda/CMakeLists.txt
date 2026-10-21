cmake_minimum_required(VERSION 3.18)

# Defaults must be set before project(): project() fills in its own (Debug for MSVC,
# the compiler's default architecture for CUDA), after which these checks never fire
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
endif()

# Native code for both GPUs: GTX 1650 Ti (Turing, 7.5) and RTX 4070 (Ada, 8.9)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 89 CACHE STRING "CUDA architectures")
endif()

project(sub_diffusion_fit_cuda LANGUAGES CXX)

# nvcc uses the first cl.exe on PATH; the 32-bit one (from "Developer Command Prompt/PowerShell")
# crashes cudafe++ with ACCESS_VIOLATION. Checked before enabling CUDA, which is where it crashes.
if(MSVC AND CMAKE_SIZEOF_VOID_P EQUAL 4)
    message(FATAL_ERROR "32-bit MSVC environment detected (${CMAKE_CXX_COMPILER}).\n"
                        "Use the \"x64 Native Tools Command Prompt for VS\" and delete the build folder.")
endif()

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # for VS Code IntelliSense

find_package(CUDAToolkit REQUIRED)

# zlib: on Windows nvcc needs MSVC, which can't use MSYS2's zlib (its zconf.h includes
# <unistd.h>), so build zlib from source and link it statically. Static also avoids
# picking up MSYS2's zlib1.dll from PATH at run time. Elsewhere use the system zlib.
if(MSVC)
    include(FetchContent)
    FetchContent_Declare(zlib
        GIT_REPOSITORY https://github.com/madler/zlib.git
        GIT_TAG v1.3.1
        GIT_SHALLOW TRUE
        EXCLUDE_FROM_ALL)
    FetchContent_MakeAvailable(zlib)
    # zlib 1.3.1's CMake doesn't export its include dirs (zconf.h is generated in the build dir)
    target_include_directories(zlibstatic INTERFACE ${zlib_SOURCE_DIR} ${zlib_BINARY_DIR})
    add_library(ZLIB::ZLIB ALIAS zlibstatic)
else()
    find_package(ZLIB REQUIRED)
endif()

# Host-side I/O is shared with the serial version
set(SERIAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../serial_cpp)

add_executable(fit_sub_cuda
    main.cu
    ${SERIAL_DIR}/read_mri.cpp
    ${SERIAL_DIR}/save_map.cpp
)

target_include_directories(fit_sub_cuda PRIVATE ${SERIAL_DIR})
target_link_libraries(fit_sub_cuda PRIVATE CUDA::cudart ZLIB::ZLIB)

# Checks the device Mittag-Leffler function against the serial one
add_executable(test_ml
    test_ml.cu
    ${SERIAL_DIR}/mittag-leffler.cpp
)
target_include_directories(test_ml PRIVATE ${SERIAL_DIR})
target_link_libraries(test_ml PRIVATE CUDA::cudart)

# Compares the GPU fit with the serial Ceres fit on one slice (built only if Ceres is found)
find_package(Ceres QUIET)
# A MinGW (MSYS2) Ceres, found via PATH, can't be linked with MSVC
if(Ceres_FOUND AND MSVC AND Ceres_DIR MATCHES "msys64|mingw")
    message(STATUS "Ignoring MinGW Ceres at ${Ceres_DIR} (incompatible with MSVC)")
    set(Ceres_FOUND FALSE)
endif()
if(Ceres_FOUND)
    add_executable(compare_serial
        compare_serial.cu
        serial_fit.cpp
        ${SERIAL_DIR}/fit_voxel_sub_ml.cpp
        ${SERIAL_DIR}/mittag-leffler.cpp
        ${SERIAL_DIR}/read_mri.cpp
    )
    target_include_directories(compare_serial PRIVATE ${SERIAL_DIR})
    target_link_libraries(compare_serial PRIVATE CUDA::cudart Ceres::ceres ZLIB::ZLIB)
else()
    message(STATUS "Ceres not found: skipping compare_serial")
endif()
