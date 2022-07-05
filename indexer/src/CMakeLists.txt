project(fast_indexer
        DESCRIPTION "Fast Feedback Indexer"
        LANGUAGES CXX CUDA)

option(BUILD_FAST_INDEXER "Build fast indexer library" ON)

if(BUILD_FAST_INDEXER)
        add_library(fast_indexer indexer_gpu.cu indexer_gpu.h indexer.cpp indexer.h)
        target_include_directories(fast_indexer PUBLIC .)
endif(BUILD_FAST_INDEXER)
