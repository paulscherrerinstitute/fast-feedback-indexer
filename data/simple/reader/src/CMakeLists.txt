cmake_policy(SET CMP0076 NEW)
cmake_policy(SET CMP0048 NEW)

project(simple_data
        DESCRIPTION "Reader for simple data"
        VERSION 1.0.0
        LANGUAGES CXX)

option(BUILD_SIMPLE_DATA_READER "Build reader API for simple data" OFF)

if(BUILD_SIMPLE_DATA_READER)
        set(simple_data_PUB_HEADER_LIST ffbidx/simple_data.h)
        add_library(simple_data INTERFACE)
        set_target_properties(simple_data PROPERTIES
                VERSION 1.0.0)
        target_compile_features(simple_data INTERFACE cxx_std_17)
        target_sources(simple_data INTERFACE ${simple_data_PUB_HEADER_LIST})
        target_include_directories(simple_data INTERFACE .)
        install(FILES ${simple_data_PUB_HEADER_LIST}
                DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ffbidx
                COMPONENT simple_data_development)
        write_basic_package_version_file(simple_data-config.cmake
                COMPATIBILITY SameMinorVersion)
        configure_file(simple_data.pc.in
                simple_data.pc
                @ONLY)
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/simple_data.pc
                DESTINATION ${CMAKE_INSTALL_DATADIR}/ffbidx/pkgconfig
                COMPONENT simple_data_development)
endif()
