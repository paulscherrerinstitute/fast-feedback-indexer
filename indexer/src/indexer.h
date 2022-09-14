#ifndef FAST_FEEDBACK_INDEXER_H
#define FAST_FEEDBACK_INDEXER_H

// Fast feedback indexer

#include <stdexcept>
#include <utility>
#include<memory>
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
    // The coordinate arrays must be of size
    // 3*n_cells at least
    template <typename float_type=float>
    struct output final {
        float_type* x;      // x coordinates, pinned memory
        float_type* y;      // y coordinates, pinned memory
        float_type* z;      // z coordinates, pinned memory
        unsigned n_cells;   // number of unit cells
    };

    // Configuration setting for the fast feedback indexer runtime state
    // These are used to determine GPU kernel grid sizes and calculation thresholds
    // Memory must be pinned in order to be used as an argument for indexing
    template <typename float_type=float>
    struct config_runtime final {
        float_type angular_step=.02;            // step through sample space [0..pi, 0..pi] with this angular step (radians)
        float_type length_threshold=.001;      // threshold for determining equal vector length (|va| - threshold < |vb| < |va| + threshold)
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
        static constexpr type null = 0u;    // instance identity usable to denote object instance without private state
        static std::atomic<type> next;      // monotonically increasing atomic counter
    };

    // Indexer object
    //
    // Keeps persistent state, like GPU memory allocations.
    template <typename float_type=float>
    struct indexer final {
        config_persistent<float_type> cpers;    // persistent configuration
        const state_id::type state;             // object instance private state identifier

        // Initialize/reconfigure this instance according to conf
        static void init (indexer<float_type>& instance, const config_persistent<float_type>& conf);

        // Drop this instance and its private state
        static void drop (indexer<float_type>& instance);

        // Create according to c
        explicit inline indexer (const config_persistent<float_type>& c)
            : state{state_id::next.fetch_add(1u)}
        { init(*this, c); }

        // Create with default persistent config
        inline indexer ()
            : state{state_id::next.fetch_add(1u)}
        { init(*this, config_persistent<float_type>{}); }

        // Create according to other.cpers
        inline indexer (const indexer& other)
            : state{state_id::next.fetch_add(1u)}
        { init(*this, other.cpers); }

        // Reconfigure according to other.cpers
        inline indexer& operator= (const indexer& other)
        { init(*this, other.cpers); return *this; }

        // Take over others state
        inline indexer (indexer&& other)
            : state(state_id::null), cpers(std::move(other.cpers))
        {
            std::swap(const_cast<state_id::type&>(state), const_cast<state_id::type&>(other.state));
        }

        // Take over others state
        inline indexer& operator= (indexer&& other)
        {
            std::swap(cpers, other.cpers);
            std::swap(const_cast<state_id::type&>(state), const_cast<state_id::type&>(other.state));
        }

        // Drop if valid
        inline ~indexer ()
        {
            if (state != state_id::null)
                drop(*this);
        }

        // Run indexing according to conf_rt on in data, put result into out data
        // All coordinate data and the runtime config memory must be pinned
        void index (const input<float_type>& in, output<float_type>& out, const config_runtime<float_type>& conf_rt);
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
