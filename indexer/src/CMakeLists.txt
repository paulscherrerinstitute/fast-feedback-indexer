project(fast_indexer
        DESCRIPTION "Fast Feedback Indexer"
        LANGUAGES CXX CUDA)

option(BUILD_FAST_INDEXER "Build fast indexer library" ON)

if(BUILD_FAST_INDEXER)
        add_library(fast_indexer
                      indexer_gpu.cu indexer_gpu.h
                      indexer.cpp indexer.h
                      log.cc log.h
                      exception.h)
        set_target_properties(fast_indexer PROPERTIES CUDA_ARCHITECTURES "75;80")
        target_include_directories(fast_indexer PUBLIC .)
endif(BUILD_FAST_INDEXER)
