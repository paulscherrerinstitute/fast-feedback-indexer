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

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <map>
#include "indexer.h"

namespace {

    constexpr struct success_type final {} success;
    constexpr struct failure_type final {} failure;

    template <typename stream>
    [[noreturn]] stream& operator<< (stream& out, [[maybe_unused]] const success_type& data)
    {
        out.flush();
        std::exit((EXIT_SUCCESS));
    }

    template <typename stream>
    [[noreturn]] stream& operator<< (stream& out, [[maybe_unused]] const failure_type& data)
    {
        out.flush();
        std::exit((EXIT_FAILURE));
    }

} // namespace

int main (int, char**)
{
    try {

        {
            fast_feedback::indexer indexer;         // indexer object with default config
            std::cout << "indexer.state " << indexer.state << '\n';         // 1
            {
                fast_feedback::indexer indexerT{};  // indexer object with default config
                std::cout << "indexerT.state " << indexerT.state << '\n';   // 2
                fast_feedback::indexer indexer0{std::move(indexerT)};
                std::cout << "indexer0.state " << indexer0.state << '\n';   // 2
                std::cout << "indexerT.state " << indexerT.state << '\n';   // 0
                if (indexerT.state != fast_feedback::state_id::null)
                    std::cerr << "Test failed: indexerT.state != null\n" << failure;
                fast_feedback::indexer indexer1{indexer0};
                std::cout << "indexer1.state " << indexer1.state << '\n';   // 3
                std::cout << "indexer0.state " << indexer0.state << '\n';   // 2
                indexer = indexer0;
                std::cout << "indexer.state " << indexer.state << '\n';     // 1
                std::cout << "indexer0.state " << indexer0.state << '\n';   // 2
                if (indexer0.state != 2u)
                    std::cerr << "Test failed: indexer0.state != 2\n" << failure;
                indexer = std::move(indexer1);
                std::cout << "indexer.state " << indexer.state << '\n';     // 3
                std::cout << "indexer1.state " << indexer1.state << '\n';   // 1
                if (indexer1.state != 1u)
                    std::cerr << "Test failed: indexer1.state != 1\n" << failure;
            }

            if (indexer.state != 3u)
                std::cerr << "Test failed: indexer.state != 3\n" << failure;
        }

        {
            std::map<unsigned, fast_feedback::indexer<float>> idxmap;
            idxmap.emplace(0u, fast_feedback::indexer{});
            if (idxmap.erase(0u) != 1)
                std::cerr << "Test failed: no map entry was deleted\n" << failure;
        }

    } catch (std::exception& ex) {
        std::cerr << "Test failed: " << ex.what() << '\n' << failure;
    }

    std::cout << "Test OK.\n" << success;
}
