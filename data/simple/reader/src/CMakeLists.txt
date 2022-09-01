cmake_minimum_required (VERSION 3.0)
cmake_policy(SET CMP0076 NEW)

project(simple_data_reader
        DESCRIPTION "Reader API for simple data"
        LANGUAGES CXX)

option(BUILD_SIMPLE_DATA_READER "Build reader API for simple data" OFF)

if(BUILD_SIMPLE_DATA_READER)
    find_package(Eigen3 3.3 NO_MODULE)
    if(NOT ${Eigen3_FOUND})
        message(FATAL_ERROR
            "Eigen3 library not found! Install a distro specific package, or use 'git submodule update' to download the submodule."
        )
    endif()

    add_library(simple_data INTERFACE)
    target_compile_features(simple_data INTERFACE cxx_std_17)
    target_sources(simple_data INTERFACE simple_data.h)
    target_include_directories(simple_data INTERFACE .)
    target_link_libraries(simple_data INTERFACE Eigen3::Eigen)
endif()
