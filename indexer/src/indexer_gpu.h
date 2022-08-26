#ifndef INDEXER_GPU_H
#define INDEXER_GPU_H

#include "indexer.h"

namespace gpu {
    using namespace fast_feedback;

    // GPU part of indexer::init
    template <typename float_type>
    void init (const indexer<float_type>& instance);

    // GPU part of indexer::drop
    template <typename float_type>
    void drop (const indexer<float_type>& instance);

    // GPU part of indexer::index
    template <typename float_type>
    void index (const indexer<float_type>& instance, const input<float_type>& in, output<float_type>& out, const config_runtime<float_type>& conf_rt);

    // Register host memory
    void pin_memory(void* ptr, std::size_t size);

    // Unregister host memory
    void unpin_memory(void* ptr);

    // Allocate pinned host memory
    void* alloc_pinned(std::size_t num_bytes);

    // Deallocate pinned host memory
    void dealloc_pinned(void* ptr);

} // namespace gpu

#endif
