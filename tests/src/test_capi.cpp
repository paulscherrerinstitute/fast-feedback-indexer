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
#include <vector>
#include <cmath>
#include "ffbidx/exception.h"
#include "ffbidx/simple_data.h"
#include "ffbidx/c_api.h"

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

int main (int argc, char *argv[])
{
    using namespace simple_data;

    try {
        if (argc <= 1)
            throw std::runtime_error("missing file argument");

        SimpleData<float, raise> data(argv[1]);         // read simple data file

        std::vector<float> x(data.spots.size() + 3);    // coordinate containers
        std::vector<float> y(data.spots.size() + 3);
        std::vector<float> z(data.spots.size() + 3);
        unsigned i=0;
        for (const auto& coord : data.unit_cell) {      // copy cell coordinates
            x[i] = coord.x;
            y[i] = coord.y;
            z[i] = coord.z;
            std::cout << "input" << i << ": " << x[i] << ", " << y[i] << ", " << z[i] << '\n';
            i++;
        }
        for (const auto& coord : data.spots) {          // copy spot coordinates
            x[i] = coord.x;
            y[i] = coord.y;
            z[i] = coord.z;
            i++;            
        }

        std::array<float, 20> buf;                      // 2x output coordinate/score container

        config_persistent cpers;                        // persistent config
        config_runtime crt;                             // runtime config
        config_ifssr cifssr;                            // ifssr refinement config
        set_defaults(&cpers, &crt, &cifssr);            // fill in defaults

        input in = {{&x[0], &y[0], &z[0]}, {&x[3], &y[3], &z[3]}, 1u, i-3u, true, true}; // indexer input object
        output out = {&buf[0], &buf[3], &buf[6], &buf[9], 1u}; // indexer output object

        std::vector<char> error_msg(128);               // memory for error message
        error err = { error_msg.data(), (unsigned)error_msg.size() };

        int h = create_indexer(&cpers, &err, nullptr);  // indexer
        if (h == -1)
            throw std::runtime_error{error_msg.data()};
        
        int idx = indexer_op(h, &in, &out, &crt, &cifssr); // run indexer
        if (idx == -1)
            throw std::runtime_error{error_msg.data()};

        int res = drop_indexer(h);                      // destruct indexer
        if (res == -1)
            throw std::runtime_error{error_msg.data()};

        const float max_score = cifssr.max_distance;    // maximum acceptable output score
        std::cout << "score: " << out.score[0];
        if (out.score[0] > max_score) {
            std::cout << " > " << max_score << " => bad\n";
            throw std::runtime_error("output score above maximum acceptable");
        } else {
            std::cout << " <= " << max_score << " => ok\n";
        }
        
        constexpr float delta = .2f;                    // it's a match if spot indices are all around delta from an integer
        constexpr unsigned n_matches = 100u;            // accept output cell if it matches so many spots
        unsigned spots_matched = 0u;

        for (const auto& spot : data.spots) {          // check for spots that match
            std::array<float, 3> m{};
            for (unsigned i=0; i<3u; i++)
                m[i] = spot.x * out.x[0] + spot.y * out.x[1] + spot.z * out.x[2];
            std::cout << "spot: " << spot.x << ' ' << spot.y << ' ' << spot.z << " --> " << m[0] << ' ' << m[1] << ' ' << m[2]; 
            int i = 0;
            while (i<3) {
                if (std::abs(m[i] - std::round(m[i])) > delta)
                    break;
                i++;
            }
            if (i == 3) {
                spots_matched++;
                std::cout << " match\n";
            } else {
                std::cout << '\n';
            }
        }

        std::cout << "=> " << spots_matched << " matches.\n";

        if (spots_matched >= n_matches)
            std::cout << "Test OK.\n" << success;

        std::cout << "Test failed.\n" << failure;

    } catch (fast_feedback::exception& ex) {
        std::cerr << "Test failed: " << ex.what() << " at " << ex.file_name << ':' << ex.line_number << '\n' << failure;
    } catch (std::exception& ex) {
        std::cerr << "Test failed: " << ex.what() << '\n' << failure;
    }
}
