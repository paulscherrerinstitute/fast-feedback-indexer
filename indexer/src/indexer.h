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

#ifndef FAST_FEEDBACK_INDEXER_H
#define FAST_FEEDBACK_INDEXER_H

// Fast feedback indexer

#include <stdexcept>
#include <utility>
#include <memory>
#include <atomic>

namespace fast_feedback {

    // Input data for fast feedback indexer
    //
    // Input data consists of the (x,y,z) reciprocal space
    // coordinates
    // - of given unit cells        [0..3*n_cells[
    // - of the spots               [3*n_cells..3*n_cells+n_spots[
    template <typename float_type=float>
    struct input final {
        float_type* x;      // x coordinates, pinned memory
        float_type* y;      // y coordinates, pinned memory
        float_type* z;      // z coordinates, pinned memory
        unsigned n_cells;   // number of given unit cells (must be before n_spots in memory, see copy_in())
        unsigned n_spots;   // number of spots (must be after n_cells in memory, see copy_in())
    };

    // Output data for fast feedback indexer
    //
    // Output data consists of the (x,y,z) reciprocal space
    // coordinates of the found unit cell vectors [0..3*n_cells[
    //
    // The coordinate arrays must be of size 3*n_cells at least,
    // the score array at least of size n_cells
    template <typename float_type=float>
    struct output final {
        float_type* x;      // x coordinates, pinned memory
        float_type* y;      // y coordinates, pinned memory
        float_type* z;      // z coordinates, pinned memory
        float_type* score;  // per cell score, pinned memory
        unsigned n_cells=1u;// number of unit cells
    };

    // Configuration setting for the fast feedback indexer runtime state
    // These are used to determine GPU kernel grid sizes and calculation thresholds
    // Memory must be pinned in order to be used as an argument for indexing
    template <typename float_type=float>
    struct config_runtime final {
        float_type length_threshold=1e-9;   // threshold for determining equal vector length (|va| - threshold < |vb| < |va| + threshold)
        float_type triml=0.01;              // lower trim value for distance to nearest integer objective value - 0 < triml < trimh
        float_type trimh=0.4;               // higher trim value for distance to nearest integer objective value - triml < trimh < 0.5
        unsigned num_sample_points=100000u; // number of sample points on half sphere for finding vector candidates
    };

    // Configuration setting for the fast feedback indexer persistent state
    // The persistent state determines static GPU memory consumtion
    // Changing these parameters will cause reallocation of memory on the GPU
    template <typename float_type=float>
    struct config_persistent final {
        unsigned max_output_cells=1;        // maximum number of output unit cells
        unsigned max_input_cells=1;         // maximum number of input unit cells, (must be before max_spots in memory, see copy_in())
        unsigned max_spots=200;             // maximum number of input spots, (must be after max_input_cells in memory, see copy_in())
        unsigned num_candidate_vectors=30;  // number of candidate vectors (per input cell vector)
    };

    // Exception type for fast feedback indexer
    struct indexer_error final : public std::runtime_error {};

    // State ID to identify object private state
    struct state_id final {
        using type = unsigned;
        static constexpr type null = 0u;                    // instance identity usable to denote object instance without private state
        static inline std::atomic<type> next = null + 1u;   // monotonically increasing atomic counter

        // Allocate a new state identifier
        static type alloc() noexcept
        {
            return next.fetch_add(1u);
        }
    };

    template <typename float_type=float> struct indexer;    // Forward declaration

    // Future to retrieve asynchronous result
    // NOTE: Make sure the data underlying the pointers survives this object
    template <typename float_type=float>
    struct future final {
        indexer<float_type>* idx;
        const input<float_type>* in;
        output<float_type>* out;
        const config_runtime<float_type>* crt;
        bool ready = false;

        inline future (indexer<float_type>& indexer,
                       const input<float_type>& input,
                       output<float_type>& output,
                       const config_runtime<float_type>& conf_rt)
            : idx{&indexer}, in{&input}, out{&output}, crt{&conf_rt}
        {}

        ~future () = default;
        future (const future&) = default;
        future& operator= (const future&) = default;
        future (future&&) = default;
        future& operator= (future&&) = default;

        // Is output data ready?
        bool is_ready ();

        // Wait for ready output data.
        void wait_for ();
    };

    // Indexer object
    //
    // Keeps persistent state, like GPU memory allocations.
    template <typename float_type>
    struct indexer final {
        config_persistent<float_type> cpers;    // persistent configuration
        const state_id::type state;             // object instance private state identifier

        // Initialize this instance according to conf
        static void init (indexer<float_type>& instance, const config_persistent<float_type>& conf);

        // Drop this instance and its private state
        static void drop (indexer<float_type>& instance);

        // Create according to c
        explicit inline indexer (const config_persistent<float_type>& c)
            : state{state_id::alloc()}
        { init(*this, c); }

        // Create with default persistent config
        inline indexer ()
            : state{state_id::alloc()}
        { init(*this, config_persistent<float_type>{}); }

        // Create according to other.cpers
        inline indexer (const indexer& other)
            : state{state_id::alloc()}
        { init(*this, other.cpers); }

        // Reconfigure according to other.cpers
        inline indexer& operator= (const indexer& other)
        { drop(*this); init(*this, other.cpers); return *this; }

        // Take over others state
        inline indexer (indexer&& other)
            : cpers(std::move(other.cpers)), state(state_id::null)
        {
            std::swap(const_cast<state_id::type&>(state), const_cast<state_id::type&>(other.state));
        }

        // Take over others state
        inline indexer& operator= (indexer&& other)
        {
            std::swap(cpers, other.cpers);
            std::swap(const_cast<state_id::type&>(state), const_cast<state_id::type&>(other.state));
            return *this;
        }

        // Drop if valid
        inline ~indexer ()
        {
            if (state != state_id::null) {
                drop(*this);
                const_cast<state_id::type&>(state) = state_id::null;
            }
        }

        // Run indexing asynchronously according to conf_rt on in data.
        // All coordinate data and the runtime config memory must be pinned
        future<float_type> index_async (const input<float_type>& in, output<float_type>& out, const config_runtime<float_type>& conf_rt);

        // Run indexing according to conf_rt on in data, put result into out data
        // All coordinate data and the runtime config memory must be pinned
        inline void index (const input<float_type>& in, output<float_type>& out, const config_runtime<float_type>& conf_rt)
        {
            index_async(in, out, conf_rt).wait_for();
        }
    };

    // Pin allocated memory during the lifetime of this object
    // Use this for already allocated unpinned memory
    // Watch out when pinning memory that might move (like container data)
    struct memory_pin final {
        void* ptr;

        // Nothing is pinned by default
        inline memory_pin()
            : ptr(nullptr)
        {}

        // Pin standard container content
        // Make sure the container data is not moved during the lifetime of the pin
        template<typename Container>
        explicit inline memory_pin(const Container& container)
            : ptr(nullptr)
        {
            void* mem_ptr = const_cast<Container&>(container).data();
            pin(mem_ptr, container.size() * sizeof(typename Container::value_type));
            ptr = mem_ptr;
        }

        // Raw memory pin
        inline memory_pin(void* mem_ptr, std::size_t num_bytes)
            : ptr(nullptr)
        {
            pin(mem_ptr, num_bytes);
            ptr = mem_ptr;
        }

        // Take over pin from other
        inline memory_pin(memory_pin&& other) noexcept
            : ptr(nullptr)
        {
            std::swap(ptr, other.ptr);
        }

        // Take over pin from other
        inline memory_pin& operator=(memory_pin&& other)
        {
            void* mem_ptr = nullptr;
            std::swap(ptr, mem_ptr);
            if (mem_ptr != nullptr)
                unpin(mem_ptr);
            std::swap(ptr, other.ptr);
            return *this;
        }

        // Unpin pinned memory if any
        inline ~memory_pin()
        {
            if (ptr != nullptr) {
                unpin(ptr);
                ptr = nullptr;
            }
        }

        memory_pin(const memory_pin&) = delete;
        memory_pin& operator=(const memory_pin&) = delete;

        // Pin on object
        // Only works for objects without internally allocated extra memory
        template<typename Object>
        static inline memory_pin on(const Object& obj)
        {
            void* mem_ptr = const_cast<Object*>(&obj);
            return memory_pin(mem_ptr, sizeof(obj));
        }

        // Pin on object underlying the pointer
        // Only works for objects without internally allocated extra memory
        template<typename Object>
        static inline memory_pin on(const Object* obj_ptr)
        {
            void* mem_ptr = const_cast<Object*>(obj_ptr);
            return memory_pin(mem_ptr, sizeof(*obj_ptr));
        }

        static void pin(void* ptr, std::size_t size);   // Raw memory pin
        static void unpin(void* ptr);                   // Raw unpin
    };

    // Allocate pinned raw memory
    void* alloc_pinned(std::size_t num_bytes);

    // Deallocate pinned memory
    void dealloc_pinned(void* ptr);

    // Deleter for pinned smart pointers
    // Calls destructor
    template<typename T>
    struct pinned_deleter final {
        inline void operator()(T* ptr) const
        {
            try {
                ptr->~T();
            } catch (...) {
                try {
                    dealloc_pinned(ptr);
                } catch (...) {}    // ignore dealloc exception
                throw;
            }
            dealloc_pinned(ptr);
        }
    };

    // Pinned smart pointer
    // Use this for memory allocated with alloc_pinned
    template<typename T>
    using pinned_ptr = std::unique_ptr<T, pinned_deleter<T>>;

    // Allocate pinned object
    // Calls default constructor
    // This only works for objects that do not allocate internal extra memory
    template<typename T>
    inline pinned_ptr<T> alloc_pinned()
    {
        pinned_ptr<T> ptr{static_cast<T*>(alloc_pinned(sizeof(T)))};
        new (ptr.get()) T;
        return ptr;
    }

} // namespace fast_feedback

#endif
