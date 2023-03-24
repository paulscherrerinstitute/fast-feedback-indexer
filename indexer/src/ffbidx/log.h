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

#ifndef INDEXER_LOG_H
#define INDEXER_LOG_H

#include <iostream>
#include <atomic>
#include <mutex>
#include <thread>
#include <string>

namespace fast_feedback {
    namespace logger {

        // Log levels
        constexpr unsigned l_fatal = 0u;    // Log fatal errors
        constexpr unsigned l_error = 1u;    // Log errors
        constexpr unsigned l_warn = 2u;     // Log warnings
        constexpr unsigned l_info = 3u;     // Log useful information
        constexpr unsigned l_debug = 4u;    // Log debug information
        constexpr unsigned l_undef = 10u;   // Undefined (only returned by get_init_log_level())

        // Current log level, default is l_error
        // Logging will react to changes dynamically
        inline std::atomic<unsigned> level{l_error};

        // Get log level from environment, if given
        // Parses the environment variable INDEXER_LOG_LEVEL
        // which should contain one of {"fatal", "error", "warn", "info", "debug"}
        // If the variable is undefined, l_undef is returned.
        unsigned get_init_log_level();

        // Initalize log level using read_init_log_level()
        // If the variable is undefined, the log level is not changed.
        inline void init_log_level()
        {
            unsigned new_level = get_init_log_level();
            if (new_level != l_undef)
                level.store(new_level);
        }

        // Get string representation for log level
        const char* level_to_string(unsigned log_level);

        // Check if log level is active
        template<unsigned log_level>
        inline bool level_active() noexcept
        {
            return log_level <= level.load();
        }

        // Get common log output lock
        std::mutex& lock() noexcept;

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

        // TODO: imlement thread safe lock_guard<log_level> for logger<log_level>
        // {
        //  guard = logger << (create guard) blabla << blabla;
        //  guard << blabla << blabla; (release guard)
        // }
    } // namespace logger
} // namespace fast_feedback

// Start and end locked logging conditional on logging level
// This will hold the logger::lock()
#define LOG_START(level) if (fast_feedback::logger::level_active<level>()) { std::lock_guard logger_output_lock{fast_feedback::logger::lock()}; do
#define LOG_END while(false); }

#endif // INDEXER_LOG_H
