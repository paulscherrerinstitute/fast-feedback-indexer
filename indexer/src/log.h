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

    // Logger writes to clog
    // Flushes only if log_level is l_fatal
    template<unsigned log_level>
    struct logger final {
        logger()
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
        if (log_level <= level.load())
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
        if (log_level <= level.load())
            std::clog << '(' << level_to_string(log_level) << ") ";
        if constexpr (log_level == l_fatal) {
            std::clog.flush();
        }
        return out;
    }

    // Logger objects
    inline logger<l_fatal> fatal;
    inline logger<l_error> error;
    inline logger<l_warn> warn;
    inline logger<l_info> info;
    inline logger<l_debug> debug;

}

#endif // INDEXER_LOG_H
