cmake_policy(SET CMP0076 NEW)
cmake_policy(SET CMP0048 NEW)

project(simple_data_reader
        DESCRIPTION "Reader API for simple data"
        VERSION 1.0.0
        LANGUAGES CXX)

option(BUILD_SIMPLE_DATA_READER "Build reader API for simple data" OFF)

if(BUILD_SIMPLE_DATA_READER)
    find_package(Eigen3 3.3 NO_MODULE)
    if(NOT ${Eigen3_FOUND})
        message(FATAL_ERROR
            "Eigen3 library not found! Install a distro specific package, or use 'git submodule update' to download the submodule."
        )
    endif()
    set(simple_data_PUB_HEADER_LIST simple_data.h)
    add_library(simple_data INTERFACE)
    set_target_properties(simple_data PROPERTIES
            VERSION 1.0.0)
    target_compile_features(simple_data INTERFACE cxx_std_17)
    target_sources(simple_data INTERFACE ${simple_data_PUB_HEADER_LIST})
    target_include_directories(simple_data INTERFACE .)
    target_link_libraries(simple_data INTERFACE Eigen3::Eigen)
    install(FILES ${simple_data_PUB_HEADER_LIST}
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ffbidx
            COMPONENT simple_data_development)
    write_basic_package_version_file(simple_data_reader-config.cmake
            COMPATIBILITY SameMinorVersion)
endif()
