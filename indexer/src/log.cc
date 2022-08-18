#include <map>
#include <string>
#include <iostream>
#include <sstream>
#include "log.h"
#include "exception.h"

namespace {

    constexpr char INDEXER_LOG_LEVEL[] = "INDEXER_LOG_LEVEL";

} // namespace

namespace logger {

    void init_log_level()
    {
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
            level = entry->second;
        }
    }

    const char* level_to_string(unsigned log_level)
    {
        switch (level) {
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

} // namespace log
