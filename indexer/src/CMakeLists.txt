cmake_policy(SET CMP0048 NEW)

project(fast_indexer
        DESCRIPTION "Library for fast feedback indexing using brute force sampling"
        VERSION 0.1.0
        LANGUAGES CXX CUDA)

option(BUILD_FAST_INDEXER "Build fast indexer library" ON)

if(BUILD_FAST_INDEXER)
        find_package(Eigen3 3.3 NO_MODULE)
        if(NOT ${Eigen3_FOUND})
                message(FATAL_ERROR
                        "Eigen3 library not found! Install a distro specific package, or use 'git submodule update' to download the submodule.")
        endif()
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
                set(CMAKE_CUDA_ARCHITECTURES "75;80")
        endif()
        find_package(CUDAToolkit REQUIRED)
        set(fast_indexer_PUB_HEADER_LIST
                indexer.h
                refine.h
                log.h
                exception.h)
        add_library(fast_indexer SHARED
                      indexer_gpu.cu indexer_gpu.h
                      indexer.cpp
                      log.cpp
                      ${fast_indexer_PUB_HEADER_LIST})
        set_target_properties(fast_indexer PROPERTIES
                                CMAKE_CUDA_KNOWN_FEATURES cuda_std_17
                                SOVERSION 0
                                VERSION 0.1.0
                                POSITION_INDEPENDENT_CODE ON)
        target_compile_features(fast_indexer PUBLIC cxx_std_17)
        target_include_directories(fast_indexer PUBLIC .)
        target_link_libraries(fast_indexer
                PRIVATE CUDA::cudart
                INTERFACE Eigen3::Eigen)
        install(TARGETS fast_indexer
                LIBRARY
                DESTINATION ${CMAKE_INSTALL_LIBDIR}
                COMPONENT ffbidx_libraries
                NAMELINK_COMPONENT ffbidx_development)
        install(FILES ${fast_indexer_PUB_HEADER_LIST}
                DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ffbidx
                COMPONENT ffbidx_development)
        cmake_path(ABSOLUTE_PATH CMAKE_INSTALL_LIBDIR
                BASE_DIRECTORY ${CMAKE_INSTALL_PREFIX}
                NORMALIZE
                OUTPUT_VARIABLE ffbidx_LIBRARY_PATH)
        install(CODE "MESSAGE(\"use LD_LIBRARY_PATH=${ffbidx_LIBRARY_PATH}\")"
                COMPONENT ffbidx_libraries)
        write_basic_package_version_file(fast_indexer-config.cmake
                COMPATIBILITY SameMinorVersion)
        configure_file(fast_indexer.pc.in
                fast_indexer.pc
                @ONLY)
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/fast_indexer.pc
                DESTINATION ${CMAKE_INSTALL_DATADIR}/ffbidx/pkgconfig
                COMPONENT ffbidx_development)
endif(BUILD_FAST_INDEXER)
