cmake_minimum_required(VERSION 3.16)
project(sub_diffusion_fit CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # for VS Code IntelliSense

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Ceres REQUIRED)
find_package(ZLIB REQUIRED)

add_executable(fit_sub
    main.cpp
    fit_voxel_sub_ml.cpp
    mittag-leffler.cpp
    read_mri.cpp
    save_map.cpp
)

target_link_libraries(fit_sub PRIVATE Ceres::ceres ZLIB::ZLIB)
