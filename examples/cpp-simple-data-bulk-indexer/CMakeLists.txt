option(SIMPLE_DATA_BULK_INDEXER "Enable test executable for bulk indexing of simple data" OFF)
option(SIMPLE_DATA_BULK_INDEXER_RPATH "Set RPATH for simple data bulk indexer executable to fast indexer library installation" OFF)

if(SIMPLE_DATA_BULK_INDEXER)
        if (SIMPLE_DATA_BULK_INDEXER_RPATH)
                cmake_path(ABSOLUTE_PATH CMAKE_INSTALL_LIBDIR
                        BASE_DIRECTORY ${CMAKE_INSTALL_PREFIX}
                        NORMALIZE
                        OUTPUT_VARIABLE simple_data_bulk_indexer_RPATH)
                if (INSTALL_RELOCATABLE)
                        cmake_path(ABSOLUTE_PATH CMAKE_INSTALL_BINDIR
                                BASE_DIRECTORY ${CMAKE_INSTALL_PREFIX}
                                NORMALIZE
                                OUTPUT_VARIABLE simple_data_bulk_indexer_ABS_PATH)
                        cmake_path(RELATIVE_PATH simple_data_bulk_indexer_RPATH
                                BASE_DIRECTORY ${simple_data_bulk_indexer_ABS_PATH})
                        set(simple_data_bulk_indexer_RPATH $ORIGIN/${simple_data_bulk_indexer_RPATH})
                endif(INSTALL_RELOCATABLE)
        endif(SIMPLE_DATA_BULK_INDEXER_RPATH)

        if(NOT BUILD_SIMPLE_DATA_READER)
                message(FATAL_ERROR "SIMPLE_DATA_BULK_INDEXER needs -DBUILD_SIMPLE_DATA_READER=1 as a cmake argument")
        endif()
        if(NOT BUILD_FAST_INDEXER)
                message(FATAL_ERROR "SIMPLE_DATA_BULK_INDEXER needs -DBUILD_FAST_INDEXER=1 as a cmake argument")
        endif()
        find_package(CUDAToolkit REQUIRED)
        add_executable(simple-data-bulk-indexer simple-data-bulk-indexer.cpp)
        target_compile_features(simple-data-bulk-indexer PRIVATE cxx_std_17)
        target_link_libraries(simple-data-bulk-indexer
                PRIVATE CUDA::cudart
                PRIVATE fast_indexer
                PRIVATE simple_data)
        if (SIMPLE_DATA_BULK_INDEXER_RPATH)
                set_target_properties(simple-data-bulk-indexer PROPERTIES
                        INSTALL_RPATH ${simple_data_bulk_indexer_RPATH})
        endif(SIMPLE_DATA_BULK_INDEXER_RPATH)
        install(TARGETS simple-data-bulk-indexer
                RUNTIME
                DESTINATION ${CMAKE_INSTALL_BINDIR}
                COMPONENT ffbidx_executables)
        if (NOT fast_feedback_indexer_ENV_SETUP MATCHES "ffbidx_PATH")
                cmake_path(ABSOLUTE_PATH CMAKE_INSTALL_BINDIR
                        BASE_DIRECTORY ${CMAKE_INSTALL_PREFIX}
                        NORMALIZE
                        OUTPUT_VARIABLE ffbidx_PATH)
                install(CODE "MESSAGE(\"use PATH=${ffbidx_PATH}\")"
                        COMPONENT ffbidx_executables)
                string(APPEND fast_feedback_indexer_ENV_SETUP "ffbidx_PATH=${ffbidx_PATH};")
                set(fast_feedback_indexer_ENV_SETUP "${fast_feedback_indexer_ENV_SETUP}" CACHE INTERNAL ENV)
        endif()
endif()
