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

#ifndef FAST_FEEDBACK_ENVVAR_H
#define FAST_FEEDBACK_ENVVAR_H

#include <sstream>
#include <stdexcept>
#include <cstdlib>
#include <functional>

namespace fast_feedback {

    template <typename result_type>
    inline result_type envvar(const char *name, std::function<result_type()> func)
    {}

    // throw std::invalid_argument, if value is not accepted
    // return if_undefined() if undefined
    template <>
    inline bool envvar<bool>(const char* name, std::function<bool()> func)
    {
        static const std::vector<std::string> accepted = {"1", "true", "yes", "on", "0", "false", "no", "off"}; // [4] == "0"
        char* varval = std::getenv(name);
        if (varval != nullptr) {
            unsigned i;
            for (i=0u; i<accepted.size(); i++) {
                if (accepted[i] == varval)
                    goto found;
            }
            {
                std::ostringstream oss;
                oss << "illegal value for " << name << ": \"" << varval << "\" (use one of";
                for (const auto& s : accepted)
                    oss << " \"" << s << '\"';
                oss << ')';
                throw std::invalid_argument(oss.str());
            }
          found:
            return i < 4u;
        }
        return func();
    }

    // throw std::invalid_argument, if value is not accepted
    template <>
    inline unsigned envvar<unsigned>(const char *name, std::function<unsigned()> func)
    {
        char* varval = std::getenv(name);
        if (varval != nullptr) {
            std::istringstream iss(varval);
            unsigned val;
            if (!(iss >> val) || !iss.eof()) {
                std::ostringstream oss;
                oss << "illegal value of type unsigned for " << name << ": " << varval;
                throw std::invalid_argument(oss.str());
            }
            return val;
        }
        return func();
    }

} // namespace fast_feedback

#endif // FAST_FEEDBACK_ENVVAR_H
