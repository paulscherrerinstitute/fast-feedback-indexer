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

#ifndef INDEXER_GPU_H
#define INDEXER_GPU_H

#include "ffbidx/indexer.h"

namespace gpu {
   using namespace fast_feedback;

   // GPU part of indexer::init
   template <typename float_type>
   void init (const indexer<float_type>& instance);

   // GPU part of indexer::drop
   template <typename float_type>
   void drop (const indexer<float_type>& instance);

   // GPU part of indexer::index_start
   template <typename float_type>
   void index_start (const indexer<float_type>& instance, const input<float_type>& in, output<float_type>& out, const config_runtime<float_type>& conf_rt,
                     void(*callback)(void*), void* data);

   // GPU part of indexer::index_end
   template <typename float_type>
   void index_end (const indexer<float_type>& instance, output<float_type>& out);

   // Register host memory
   void pin_memory(void* ptr, std::size_t size);

   // Unregister host memory
   void unpin_memory(void* ptr);

   // Allocate pinned host memory
   void* alloc_pinned(std::size_t num_bytes);

   // Deallocate pinned host memory
   void dealloc_pinned(void* ptr);
   
   // GPU version of check_config
   template<typename float_type=float>
   void check_config(const config_persistent<float_type>* cpers, const config_runtime<float_type>* crt);

} // namespace gpu

#endif
