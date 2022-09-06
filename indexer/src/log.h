#ifndef INDEXER_LOG_H
#define INDEXER_LOG_H

#include <iostream>
#include <atomic>
#include <string>

namespace logger {

    // Log levels
    constexpr unsigned l_fatal = 0u;
    constexpr unsigned l_error = 1u;
    constexpr unsigned l_warn = 2u;
    constexpr unsigned l_info = 3u;
    constexpr unsigned l_debug = 4u;

    // Current log level
    // Logging will react to changes dynamically
    inline std::atomic<unsigned> level{l_error};

    // Initialize log level from environment, if given
    void init_log_level();

    // Get string representation for log level
    const char* level_to_string(unsigned log_level);

    // Check if log level is active
    template<unsigned log_level>
    inline bool level_active() noexcept
    {
        return log_level <= level.load();
    }

    // Logger writes to clog
    // Flushes only if log_level is l_fatal
    template<unsigned log_level>
    struct logger final {
        inline logger()
        {
            init_log_level();
        }
    };

    // For starting a stanza
    inline struct stanza_type final {} stanza;

    // Log output to clog
    template<typename T, unsigned log_level>
    inline logger<log_level>& operator<<(logger<log_level>& out, const T& value)
    {
        if (level_active<log_level>())
            std::clog << value;
        if constexpr (log_level == l_fatal) {
            std::clog.flush();
        }
        return out;
    }

    // Start stanza
    template<unsigned log_level>
    inline logger<log_level>& operator<<(logger<log_level>& out, const stanza_type&)
    {
        if (level_active<log_level>())
            std::clog << '(' << level_to_string(log_level) << ") ";
        if constexpr (log_level == l_fatal) {
            std::clog.flush();
        }
        return out;
    }

    #ifndef _INDEXER_LOG_IMPL_
        // Logger objects
        extern logger<l_fatal> fatal;
        extern logger<l_error> error;
        extern logger<l_warn> warn;
        extern logger<l_info> info;
        extern logger<l_debug> debug;
    #endif

}

#endif // INDEXER_LOG_H
