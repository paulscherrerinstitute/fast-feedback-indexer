/*
Copyright 2022 Paul Scherrer Institute

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.

------------------------

Author: hans-christian.stadler@psi.ch
*/

#include "exception.h"
#include "indexer.h"
#include "indexer_gpu.h"

namespace fast_feedback {

    std::atomic<state_id::type> state_id::next = state_id::null + 1u;

    template <typename float_type>
    void indexer<float_type>::init (indexer<float_type>& instance, const config_persistent<float_type>& conf)
    {
        if (instance.state == state_id::null)
            throw FF_EXCEPTION("illegal initialisation of null state");
        
        instance.cpers = conf;
        gpu::drop(instance);
        gpu::init(instance);
    }

    template <typename float_type>
    void indexer<float_type>::drop (indexer<float_type>& instance)
    {
        if (instance.state != state_id::null)
            gpu::drop(instance);
    }

    template <typename float_type>
    void indexer<float_type>::index(const input<float_type>& in, output<float_type>& out, const config_runtime<float_type>& conf_rt)
    {
        gpu::index(*this, in, out, conf_rt);
    }

    void memory_pin::pin(void* ptr, std::size_t size)
    {
        gpu::pin_memory(ptr, size);
    }

    void memory_pin::unpin(void* ptr)
    {
        gpu::unpin_memory(ptr);
    }

    void* alloc_pinned(std::size_t num_bytes)
    {
        return gpu::alloc_pinned(num_bytes);
    }

    void dealloc_pinned(void* ptr)
    {
        gpu::dealloc_pinned(ptr);
    }

    template void indexer<float>::init(indexer<float>&, const config_persistent<float>&);
    template void indexer<float>::drop(indexer<float>&);
    template void indexer<float>::index(const input<float>&, output<float>&, const config_runtime<float>&);

} // namespace fast_feedback
