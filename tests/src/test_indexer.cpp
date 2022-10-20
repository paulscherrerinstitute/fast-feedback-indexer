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
#include <array>
#include <Eigen/Core>
#include <Eigen/LU>
#include "simple_data.h"
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

        std::array<float, 10> buf;                      // output coordinate/score container
        fast_feedback::config_runtime<float> crt{};     // default runtime config
        fast_feedback::indexer indexer;                 // indexer object with default config

        fast_feedback::memory_pin pin_x{x};             // pin input coordinate containers
        fast_feedback::memory_pin pin_y{y};
        fast_feedback::memory_pin pin_z{z};
        fast_feedback::memory_pin pin_buf{buf};         // pin output coordinate container
        fast_feedback::memory_pin pin_crt{fast_feedback::memory_pin::on(crt)};  // pin runtime config memory

        fast_feedback::input<float> in{x.data(), y.data(), z.data(), 1u, i-3u}; // create indexer input object
        fast_feedback::output<float> out{&buf[0], &buf[3], &buf[6], &buf[9]};   // create indexer output object

        indexer.index(in, out, crt);                                            // run indexer

        constexpr float max_score = -200.0f;            // maximum acceptable output score
        std::cout << "score: " << out.score[0];
        if (out.score[0] > max_score) {
            std::cout << " > " << max_score << " => bad\n";
            throw std::runtime_error("output score above maximum acceptable");
        } else {
            std::cout << " <= " << max_score << " => ok\n";
        }
        
        constexpr float delta = .2f;                    // it's a match if spot indices are all around delta from an integer
        constexpr unsigned n_matches = 20u;             // accept output cell if it matches so many spots
        unsigned spots_matched = 0u;
        Eigen::Matrix<float, 3, 3> B;                   // unit cell base
        B << out.x[0], out.x[1], out.x[2],
             out.y[0], out.y[1], out.y[2],
             out.z[0], out.z[1], out.z[2];
        std::cout << "cell:\n" << B << '\n';
        Eigen::Matrix<float, 3, 1> s;                   // spot

        for (const auto& spot : data.spots) {          // check for spots that match
            s << spot.x, spot.y, spot.z;
            auto m = B.inverse() * s;
            std::cout << "spot: " << s[0] << ' ' << s[1] << ' ' << s[2] << " --> " << m[0] << ' ' << m[1] << ' ' << m[2]; 
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

        std::cout << "Test " << ( (spots_matched >= n_matches) ? "OK" : "failed" ) << ".\n";

    } catch (std::exception& ex) {
        std::cerr << "Test failed: " << ex.what() << '\n' << failure;
    }
}
