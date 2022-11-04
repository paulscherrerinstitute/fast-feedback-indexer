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

#define _INDEXER_LOG_IMPL_

#include <map>
#include <string>
#include <iostream>
#include <sstream>
#include <atomic>
#include <mutex>
#include "log.h"
#include "exception.h"

namespace {

    // Is logging initialized?
    std::atomic_bool logging_initialized = false;

    // Protect logging initalisation
    std::mutex logging_init_lock;

    constexpr char INDEXER_LOG_LEVEL[] = "INDEXER_LOG_LEVEL";

} // namespace

namespace logger {

    void init_log_level()
    {
        if (logging_initialized.load())
            return;
        
        {
            std::lock_guard<std::mutex> lock(logging_init_lock);

            if (logging_initialized.load())
                return;

            const std::map<std::string, unsigned> string_to_level = {
                {"fatal", l_fatal},
                {"error", l_error},
                {"warn", l_warn},
                {"info", l_info},
                {"debug", l_debug}
            };

            char* l_string = std::getenv(INDEXER_LOG_LEVEL);
            if (l_string != nullptr) {
                auto entry = string_to_level.find(l_string);
                if (entry == string_to_level.end()) {
                    std::ostringstream oss;
                    for (const auto& e : string_to_level)
                        oss << e.first << ", ";
                    std::string levels_list = oss.str();
                    levels_list.erase(levels_list.size() - 2);
                    throw FF_EXCEPTION_OBJ << "illegal value for " << INDEXER_LOG_LEVEL << ": " << l_string << " (should be in [" << levels_list << "])\n";
                }
                level.store(entry->second);
            }
        } // release logging_init_lock
    }

    const char* level_to_string(unsigned log_level)
    {
        switch (log_level) {
            case l_fatal:
                return "fatal";
            case l_error:
                return "error";
            case l_warn:
                return "warn";
            case l_info:
                return "info";
            case l_debug:
                return "debug";
        }
        
        return "undef";
    }

    logger<l_fatal> fatal;
    logger<l_error> error;
    logger<l_warn> warn;
    logger<l_info> info;
    logger<l_debug> debug;

} // namespace log