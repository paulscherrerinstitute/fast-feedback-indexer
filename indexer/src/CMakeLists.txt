cmake_policy(SET CMP0048 NEW)

project(fast_indexer
        DESCRIPTION "Fast Feedback Indexer"
        VERSION 0.1.0
        LANGUAGES CXX CUDA)

option(BUILD_FAST_INDEXER "Build fast indexer library" ON)

if(BUILD_FAST_INDEXER)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
                set(CMAKE_CUDA_ARCHITECTURES "75;80")
        endif()
        find_package(CUDAToolkit REQUIRED)
        set(fast_indexer_PUB_HEADER_LIST
                indexer.h
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
        target_link_libraries(fast_indexer PRIVATE CUDA::cudart)
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
endif(BUILD_FAST_INDEXER)
