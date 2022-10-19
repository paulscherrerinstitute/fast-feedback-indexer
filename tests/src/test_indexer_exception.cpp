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

#include <iostream>
#include <string>
#include "exception.h"

namespace {
    fast_feedback::exception create_copy()
    {
        return fast_feedback::exception(FF_EXCEPTION("12"));    // <-- this line number
    }

    fast_feedback::exception move_assign_copy()
    {
        fast_feedback::exception ex1{FF_EXCEPTION_OBJ};
        fast_feedback::exception ex2 = FF_EXCEPTION_OBJ;
        ex1 = std::move(create_copy());
        ex2 = ex1;
        return ex2;
    }
}

int main(int, char *argv[])
{
    fast_feedback::exception ex = move_assign_copy() << "34";

    std::string msg{ex.what()};
    std::string fname{ex.file_name};
    std::string failed{"Test failed: "};

    if (ex.line_number == 33) { // <-- put line number here
        if (fname.find("test_indexer_exception.cpp") != std::string::npos) {
            if (msg == "1234") {
                std::cout << "Test OK.\n";
                return ((EXIT_SUCCESS));
            } else {
                std::cerr << failed << "msg=" << msg << '\n';
            }
        } else {
            std::cerr << failed << "fname=" << fname << '\n';
        }
    } else {
        std::cerr << failed << "line=" << ex.line_number << '\n';
    }

    return ((EXIT_FAILURE));
}
