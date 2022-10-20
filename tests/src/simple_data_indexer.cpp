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
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <array>
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
        if (argc <= 3)
            throw std::runtime_error("missing arguments <file name> <number of kept candidate vectors> <number of half sphere sample points>");

        fast_feedback::config_runtime<float> crt{};         // default runtime config
        {
            std::istringstream iss(argv[3]);
            iss >> crt.num_sample_points;
            if (! iss)
                throw std::runtime_error("unable to parse second argument: number of half sphere sample points");
            std::cout << "n_samples=" << crt.num_sample_points << '\n';
        }

        fast_feedback::config_persistent<float> cpers{};    // default persistent config
        {
            std::istringstream iss(argv[2]);
            iss >> cpers.num_candidate_vectors;
            if (! iss)
                throw std::runtime_error("unable to parse second argument: number of kept candidate vectors");
            std::cout << "n_candidates=" << cpers.num_candidate_vectors << '\n';
        }

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

        std::array<float, 4*3> buf;                     // output coordinate container
        fast_feedback::indexer indexer{cpers};          // indexer object

        fast_feedback::memory_pin pin_x{x};             // pin input coordinate containers
        fast_feedback::memory_pin pin_y{y};
        fast_feedback::memory_pin pin_z{z};
        fast_feedback::memory_pin pin_buf{buf};         // pin output coordinate container
        fast_feedback::memory_pin pin_crt{fast_feedback::memory_pin::on(crt)};  // pin runtime config memory

        fast_feedback::input<float> in{x.data(), y.data(), z.data(), 1u, i-3u}; // create indexer input object
        fast_feedback::output<float> out{&buf[0], &buf[3], &buf[6], &buf[9]};   // create indexer output object

        indexer.index(in, out, crt);                                            // run indexer

        std::cout << "cell_score=" << out.score[0] << '\n';
        for (unsigned i=0u; i<3u; i++)
            std::cout << "output" << i << ": " << out.x[i] << ", " << out.y[i] << ", " << out.z[i] << '\n';

    } catch (std::exception& ex) {
        std::cerr << "indexing failed: " << ex.what() << '\n' << failure;
    }

    std::cout << success;
}
