cmake_policy(SET CMP0048 NEW)

project(fast_indexer
        DESCRIPTION "Library for fast feedback indexing using brute force sampling"
        VERSION 0.1.0
        LANGUAGES CXX CUDA)

option(BUILD_FAST_INDEXER "Build fast indexer library" ON)
option(BUILD_FAST_INDEXER_STATIC "Build fast indexer static library" ON)

if (BUILD_FAST_INDEXER OR BUILD_FAST_INDEXER_STATIC)
        set(THREADS_PREFER_PTHREAD_FLAG ON)
        find_package(Threads REQUIRED)
        find_package(Eigen3 3.4 NO_MODULE)
        if(NOT ${Eigen3_FOUND})
                message(FATAL_ERROR
                        "Eigen3 library not found! Install a distro specific package, or use 'git submodule update' to download the submodule.")
        endif()
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
                set(CMAKE_CUDA_ARCHITECTURES "75;80")
        endif()
        find_package(CUDAToolkit REQUIRED)
        message("CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
        set(fast_indexer_PUB_HEADER_LIST
                ffbidx/indexer.h
                ffbidx/refine.h
                ffbidx/log.h
                ffbidx/exception.h)
        cmake_path(ABSOLUTE_PATH CMAKE_INSTALL_LIBDIR
                BASE_DIRECTORY ${CMAKE_INSTALL_PREFIX}
                NORMALIZE
                OUTPUT_VARIABLE ffbidx_LIBRARY_PATH)
        set(ffbidx_PKG_PATH ${CMAKE_INSTALL_DATADIR}/ffbidx/pkgconfig)
        set(ffbidx_PKG_PATH_IN_LIB ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
        set(ffbidx_INCLUDE_PATH ${CMAKE_INSTALL_INCLUDEDIR})
endif()

if(BUILD_FAST_INDEXER)
        add_library(fast_indexer SHARED
                indexer_gpu.cu ffbidx/indexer_gpu.h
                indexer.cpp
                log.cpp
                ${fast_indexer_PUB_HEADER_LIST})
        set_target_properties(fast_indexer PROPERTIES
                CUDA_RUNTIME_LIBRARY Shared
                SOVERSION 0
                VERSION 0.1.0
                POSITION_INDEPENDENT_CODE ON)
        target_compile_features(fast_indexer PUBLIC cxx_std_17)
        target_include_directories(fast_indexer PUBLIC .)
        target_link_libraries(fast_indexer
                PRIVATE CUDA::cudart
                PRIVATE Threads::Threads
                INTERFACE Eigen3::Eigen)
        install(TARGETS fast_indexer
                LIBRARY
                DESTINATION ${CMAKE_INSTALL_LIBDIR}
                COMPONENT ffbidx_libraries
                NAMELINK_COMPONENT ffbidx_development)
        install(CODE "MESSAGE(\"use LD_LIBRARY_PATH=${ffbidx_LIBRARY_PATH}\")"
                COMPONENT ffbidx_libraries)
        configure_file(fast_indexer.pc.in
                fast_indexer.pc
                @ONLY)
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/fast_indexer.pc
                DESTINATION ${ffbidx_PKG_PATH_IN_LIB}
                COMPONENT ffbidx_development_extra)
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/fast_indexer.pc
                DESTINATION ${ffbidx_PKG_PATH}
                COMPONENT ffbidx_development)
endif(BUILD_FAST_INDEXER)

if(BUILD_FAST_INDEXER_STATIC)
        add_library(fast_indexer_static STATIC
                indexer_gpu.cu ffbidx/indexer_gpu.h
                indexer.cpp
                log.cpp
                ${fast_indexer_PUB_HEADER_LIST})
        set_target_properties(fast_indexer_static PROPERTIES
                CUDA_RUNTIME_LIBRARY Static
                VERSION 0.1.0)
        target_compile_features(fast_indexer_static PUBLIC cxx_std_17)
        target_include_directories(fast_indexer_static PUBLIC .)
        target_link_libraries(fast_indexer_static
                PRIVATE CUDA::cudart_static
                PRIVATE Threads::Threads
                INTERFACE Eigen3::Eigen)
        install(TARGETS fast_indexer_static
                LIBRARY
                DESTINATION ${CMAKE_INSTALL_LIBDIR}
                COMPONENT ffbidx_static_libraries)
        configure_file(fast_indexer_static.pc.in
                fast_indexer_static.pc
                @ONLY)
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/fast_indexer_static.pc
                DESTINATION ${ffbidx_PKG_PATH_IN_LIB}
                COMPONENT ffbidx_development_extra)
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/fast_indexer_static.pc
                DESTINATION ${ffbidx_PKG_PATH}
                COMPONENT ffbidx_development)
endif(BUILD_FAST_INDEXER_STATIC)

if (BUILD_FAST_INDEXER OR BUILD_FAST_INDEXER_STATIC)
        install(FILES ${fast_indexer_PUB_HEADER_LIST}
                DESTINATION ${ffbidx_INCLUDE_PATH}/ffbidx
                COMPONENT ffbidx_development)
        cmake_path(ABSOLUTE_PATH ffbidx_INCLUDE_PATH
                BASE_DIRECTORY ${CMAKE_INSTALL_PREFIX}
                NORMALIZE
                OUTPUT_VARIABLE ffbidx_CPLUS_INCLUDE_PATH)
        cmake_path(ABSOLUTE_PATH ffbidx_PKG_PATH
                BASE_DIRECTORY ${CMAKE_INSTALL_PREFIX}
                NORMALIZE
                OUTPUT_VARIABLE ffbidx_PKG_CONFIG_PATH)
        install(CODE "MESSAGE(\"use PKG_CONFIG_PATH=${ffbidx_PKG_PATH}\")"
                COMPONENT ffbidx_libraries)
        install(CODE "MESSAGE(\"use CPLUS_INCLUDE_PATH=${ffbidx_CPLUS_INCLUDE_PATH}\")")
        install(CODE "MESSAGE(\"use LIBRARY_PATH=${ffbidx_LIBRARY_PATH}\")"
                COMPONENT ffbidx_libraries)
        string(APPEND fast_feedback_indexer_ENV_SETUP "ffbidx_LIBRARY_PATH=${ffbidx_LIBRARY_PATH};ffbidx_CPLUS_INCLUDE_PATH=${ffbidx_CPLUS_INCLUDE_PATH};ffbidx_PKG_CONFIG_PATH=${ffbidx_PKG_CONFIG_PATH};")
        set(fast_feedback_indexer_ENV_SETUP "${fast_feedback_indexer_ENV_SETUP}" CACHE INTERNAL ENV)
        write_basic_package_version_file(fast_indexer-config.cmake
                COMPATIBILITY SameMinorVersion)
endif()
