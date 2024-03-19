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

#ifndef FAST_FEEDBACK_EXCEPTION_H
#define FAST_FEEDBACK_EXCEPTION_H

#include <exception>
#include <sstream>
#include "envvar.h"

namespace fast_feedback {

    constexpr char INDEXER_VERBOSE_EXCEPTION[] = "INDEXER_VERBOSE_EXCEPTION";

    struct exception : public std::exception {

        std::string error_message;
        const char* file_name;
        unsigned line_number;

        inline static bool verbose()
        {
            return envvar<bool>(INDEXER_VERBOSE_EXCEPTION, []()->bool{return false;});
        }

        inline static std::string verbosify(const std::string& message, const char* file, unsigned line)
        {
            if (verbose()) {
                std::ostringstream oss;
                oss << '(' << file << ':' << line << ") " << message;
                return oss.str();
            }
            return message;
        }

        inline exception(const std::string& message, const char* file, unsigned line)
            : error_message(verbosify(message, file, line)), file_name(file), line_number(line)
        {}

        inline exception(const exception&) = default;
        inline exception(exception&&) = default;

        inline exception& operator=(const exception&) = default;
        inline exception& operator=(exception&&) = default;

        inline ~exception() noexcept override
        {}

        inline const char* what() const noexcept override
        {
            return error_message.c_str();
        }

        template<typename T>
        inline exception& operator<<(const T& data)
        {
            std::ostringstream oss;
            oss << data;
            error_message += oss.str();
            return *this;
        }

    };

}

#define FF_EXCEPTION(msg) fast_feedback::exception(msg, __FILE__, __LINE__)
#define FF_EXCEPTION_OBJ fast_feedback::exception{std::string{}, __FILE__, __LINE__}

#endif
