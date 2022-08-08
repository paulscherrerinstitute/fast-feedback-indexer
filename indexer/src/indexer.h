#ifndef FAST_FEEDBACK_INDEXER_H
#define FAST_FEEDBACK_INDEXER_H

// Fast feedback indexer

#include <stdexcept>
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
        float_type* x;      // x coordinates
        float_type* y;      // y coordinates
        float_type* z;      // z coordinates
        unsigned n_cells;   // number of given unit cells
        unsigned n_spots;   // number of spots
    };

    // Output data for fast feedback indexer
    //
    // Output data consists of the (x,y,z) reciprocal space
    // coordinates of the found unit cell vectors [0..3*n_cells-1]
    //
    // The coordinate arrays must be of size
    // 3*n_cells at least
    template <typename float_type=float>
    struct output final {
        float_type* x;      // x coordinates
        float_type* y;      // y coordinates
        float_type* z;      // z coordinates
        unsigned n_cells;   // number of unit cells
    };

    // Configuration setting for the fast feedback indexer runtime state
    template <typename float_type=float>
    struct config_runtime final {
        float_type angular_step=.02;    // step through sample space [0..pi, 0..pi] with this angular step (radians)
    };

    // Configuration setting for the fast feedback indexer persistent state
    template <typename float_type=float>
    struct config_persistent final {
        unsigned max_output_cells=1;    // maximum number of output unit cells
        unsigned max_input_cells=1;     // maximum number of input unit cells
        unsigned max_spots=200;         // maximum number of input spots
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
    // May keep some persistent state, like GPU memory allocations.
    template <typename float_type=float>
    struct indexer final {
        config_persistent<float_type> cpers;    // persistent configuration
        const state_id::type state;             // object instance private state identifier

        // Initialize/reconfigure this instance to persistent state conf
        static void init (indexer<float_type>& instance, const config_persistent<float_type>& conf);

        // Drop this instance and its private state
        static void drop (indexer<float_type>& instance);

        inline indexer (const config_persistent<float_type>& c)
            : state{state_id::next.fetch_add(1u)}
        { init(*this, c); }

        inline indexer ()
            : state{state_id::next.fetch_add(1u)}
        { init(*this, config_persistent<float_type>{}); }

        inline indexer (const indexer& other)
            : state{state_id::next.fetch_add(1u)}
        { init(*this, other.cpers); }

        inline indexer& operator= (const indexer& other)
        { init(*this, other.cpers); return *this; }

        inline indexer (indexer&& other)
            : cpers{std::move(other.cpers)}, state{other.state}
        {
            const_cast<state_id::type&>(other.state) = state_id::null;
        }

        inline indexer& operator= (indexer&& other)
        {
            cpers = std::move(other.cpers);
            state = other.state;
            const_cast<state_id::type&>(other.state) = state_id::null;
        }

        inline ~indexer ()
        {
            if (state != state_id::null)
                drop(*this);
        }

        void index (const input<float_type>& in, output<float_type>& out, const config_runtime<float_type>& conf_rt);
    };

} // namespace fast_feedback

#endif
