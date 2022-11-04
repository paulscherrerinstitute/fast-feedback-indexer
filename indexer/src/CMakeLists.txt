project(fast_indexer
        DESCRIPTION "Fast Feedback Indexer"
        LANGUAGES CXX CUDA)

option(BUILD_FAST_INDEXER "Build fast indexer library" ON)

if(BUILD_FAST_INDEXER)
        find_package(CUDAToolkit REQUIRED)
        add_library(fast_indexer
                      indexer_gpu.cu indexer_gpu.h
                      indexer.cpp indexer.h
                      log.cpp log.h
                      exception.h)
        set_target_properties(fast_indexer PROPERTIES CUDA_ARCHITECTURES "75;80")
        target_compile_features(fast_indexer PUBLIC cxx_std_17)
        target_include_directories(fast_indexer PUBLIC .)
        target_link_libraries(fast_indexer PRIVATE CUDA::cudart)
        set_property(TARGET fast_indexer PROPERTY POSITION_INDEPENDENT_CODE ON)
endif(BUILD_FAST_INDEXER)
