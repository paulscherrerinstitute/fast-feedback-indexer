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
#include <sstream>
#include <stdexcept>
#include <chrono>
#include <Eigen/Dense>
#include "simple_data.h"
#include "refine.h"

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
    using clock = std::chrono::high_resolution_clock;
    using duration = std::chrono::duration<double>;

    try {
        if (argc <= 8)
            throw std::runtime_error("missing arguments, use\n<file name> <max number of spots> <max number of output cells> <number of kept candidate vectors> <number of half sphere sample points> ((lsq <lsq spot filter threshold> <score filter threshold>) | (ifme <error sensitivity> <number of iterations>))");

        fast_feedback::config_runtime<float> crt{};         // default runtime config
        {
            std::istringstream iss(argv[5]);
            iss >> crt.num_sample_points;
            if (! iss)
                throw std::runtime_error("unable to parse fifth argument: number of half sphere sample points");
            std::cout << "n_samples=" << crt.num_sample_points << '\n';
        }

        fast_feedback::config_persistent<float> cpers{};    // default persistent config
        {
            {
                std::istringstream iss(argv[2]);
                iss >> cpers.max_spots;
                if (! iss)
                    throw std::runtime_error("unable to parse second argument: max number of spots");
                std::cout << "max_spots=" << cpers.max_spots << '\n';
            }
            {
                std::istringstream iss(argv[3]);
                iss >> cpers.max_output_cells;
                if (! iss)
                    throw std::runtime_error("unable to parse third argument: max number of output cells");
                std::cout << "max_output_cells=" << cpers.max_output_cells << '\n';
            }
            {
                std::istringstream iss(argv[4]);
                iss >> cpers.num_candidate_vectors;
                if (! iss)
                    throw std::runtime_error("unable to parse fourth argument: number of kept candidate vectors");
                std::cout << "n_candidates=" << cpers.num_candidate_vectors << '\n';
            }
        }

        fast_feedback::refine::indexer<float>* indexer_p = nullptr;
        std::string method{argv[6]};
        std::cout << "method=" << method << '\n';

        SimpleData<float, raise> data{argv[1]};         // read simple data file

        auto t0 = clock::now();

        if (method == "lsq") {
            fast_feedback::refine::config_lsq clsq{};   // default lsq refinement config
            {
                {
                    std::istringstream iss(argv[7]);
                    iss >> clsq.fit_threshold;
                    if (! iss)
                        throw std::runtime_error("unable to parse sixth argument: threshold for spot selection in least squares fitting");
                    std::cout << "lsq_threshold=" << clsq.fit_threshold << '\n';
                    if ((clsq.fit_threshold <= .0f) || (clsq.fit_threshold > .5f))
                        throw std::runtime_error("lsq_threshold must be in range (0..0.5]");
                }
                {
                    std::istringstream iss(argv[8]);
                    iss >> clsq.score_threshold;
                    if (! iss)
                        throw std::runtime_error("unable to parse seventh argument: threshold for spot selection in score calculation");
                    std::cout << "score_threshold=" << clsq.score_threshold << '\n';
                    if ((clsq.score_threshold <= .0f) || (clsq.score_threshold > .5f))
                        throw std::runtime_error("score_threshold must be in range (0..0.5]");
                }
            }
            indexer_p = new fast_feedback::refine::indexer_lsq{cpers, crt, clsq};
        } else if (method == "ifme") {
            fast_feedback::refine::config_ifme cifme{}; // default ifme refinement config
            {
                {
                    std::istringstream iss(argv[7]);
                    iss >> cifme.error_sensitivity;
                    if (! iss)
                        throw std::runtime_error("unable to parse sixth argument: error sensitivity for iterated fit to modified errors");
                    std::cout << "error_sensitivity=" << cifme.error_sensitivity << '\n';
                    if (cifme.error_sensitivity < .0f)
                        throw std::runtime_error("error_sensitivity must be greter equal to 0");
                }
                {
                    std::istringstream iss(argv[8]);
                    iss >> cifme.n_iter;
                    if (! iss)
                        throw std::runtime_error("unable to parse seventh argument: number of iterations for iterated fit to modified errors");
                    std::cout << "n_iter=" << cifme.n_iter << '\n';
                    if (cifme.n_iter <= .0f)
                        throw std::runtime_error("number of iterations must be bigger than 0");
                }
            }
            indexer_p = new fast_feedback::refine::indexer_ifme{cpers, crt, cifme};
        } else {
            throw std::runtime_error("indexer method must be one of 'lsq', 'ifme'");
        }

        fast_feedback::refine::indexer<float>& indexer = *indexer_p;

        unsigned i=0u;
        for (const auto& coord : data.unit_cell) {      // copy cell coordinates
            indexer.iCellX(0, i) = coord.x;
            indexer.iCellY(0, i) = coord.y;
            indexer.iCellZ(0, i) = coord.z;
            i++;
        }
        if (i != 3u)
            throw std::logic_error("simple data contains more than one input cell");
        std::cout << "input:\n" << indexer.iCellM() << '\n';
        i=0;
        for (const auto& coord : data.spots) {          // copy spot coordinates
            indexer.spotX(i) = coord.x;
            indexer.spotY(i) = coord.y;
            indexer.spotZ(i) = coord.z;
            if (++i == indexer.max_spots())
                break;
        }

        auto t1 = clock::now();
        indexer.index(1u, i);                           // run refined indexing
        auto t2 = clock::now();

        std::cout.precision(12);
        std::cout << "output:\n";
        for (unsigned j=0u; j<cpers.max_output_cells; j++)
            std::cout << indexer.oCellM().block(3u * j, 0u, 3u, 3u) << "\n\n";
        std::cout << "scores:\n" << indexer.oScoreV() << '\n';
        std::cout << "timings:\n"
                  << "prep    " << duration{t1 - t0}.count() << "s\n"
                  << "index   " << duration{t2 - t1}.count() << "s\n";

    } catch (std::exception& ex) {
        std::cerr << "indexing failed: " << ex.what() << '\n' << failure;
    }

    std::cout << success;
}
