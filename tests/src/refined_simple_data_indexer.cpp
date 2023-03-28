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
#include "ffbidx/simple_data.h"
#include "ffbidx/refine.h"

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

    template<typename config_type>
    void parse_conf(config_type& conf, char* argv[])
    {
        {
            std::istringstream iss(argv[7]);
            iss >> conf.threshold_contraction;
            if (! iss)
                throw std::runtime_error("unable to parse sixth argument: threshold contraction");
            std::cout << "threshold_contraction=" << conf.threshold_contraction << '\n';
            if ((conf.threshold_contraction <= .0f) || (conf.threshold_contraction >= 1.f))
                throw std::runtime_error("threshold_contraction must be in range (0..1)");
        }
        {
            std::istringstream iss(argv[8]);
            iss >> conf.min_spots;
            if (! iss)
                throw std::runtime_error("unable to parse seventh argument: minimum number of spots for fitting");
            std::cout << "min_spots=" << conf.min_spots << '\n';
            if (conf.min_spots <= 3u)
                throw std::runtime_error("min_spots must be > 3");
        }
        {
            std::istringstream iss(argv[9]);
            iss >> conf.max_iter;
            if (! iss)
                throw std::runtime_error("unable to parse eight argument: max iterations");
            std::cout << "max_iter=" << conf.max_iter << '\n';
            if (conf.max_iter <= 0)
                throw std::runtime_error("max iterations must be positive");
        }        
    }

} // namespace

int main (int argc, char *argv[])
{
    using namespace simple_data;
    using clock = std::chrono::high_resolution_clock;
    using duration = std::chrono::duration<double>;

    try {
        if ((argc <= 8) || ((std::string{argv[6]} == "ifse") && (argc <= 9)))
            throw std::runtime_error("missing arguments, use\n<file name> <max number of spots> <max number of output cells> <number of kept candidate vectors> <number of half sphere sample points> ((ifss <threshold contraction> <min spots>) | (ifse <contraction speed> <min spots> <max iterations>))");

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

        if (method == "ifss") {
            fast_feedback::refine::config_ifss cifss{};   // default ifss refinement config
            parse_conf(cifss, argv);
            indexer_p = new fast_feedback::refine::indexer_ifss{cpers, crt, cifss};
        } else if (method == "ifse") {
            fast_feedback::refine::config_ifse cifse{}; // default ifse refinement config
            parse_conf(cifse, argv);
            indexer_p = new fast_feedback::refine::indexer_ifse{cpers, crt, cifse};
        } else {
            throw std::runtime_error("indexer method must be one of 'ifss', 'ifse'");
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
            std::cout << indexer.oCell(j) << "\n\n";
        std::cout << "scores:\n" << indexer.oScoreV() << "\n\n";
        unsigned best_cell = fast_feedback::refine::best_cell(indexer.oScoreV());
        bool indexable = fast_feedback::refine::is_viable_cell(indexer.oCell(best_cell), indexer.Spots());
        std::vector<unsigned> crystalls = fast_feedback::refine::compute_crystalls(indexer.oCellM(), indexer.Spots(), indexer.oScoreV());
        std::cout << "best cell: " << best_cell << ", is viable: " << (indexable ? "true " : "false") << '\n';
        std::cout << "crystalls:\n" << Eigen::Map<Eigen::VectorX<unsigned>>(crystalls.data(), crystalls.size()) << "\n\n";
        std::cout << "timings:\n"
                  << "prep    " << duration{t1 - t0}.count() << "s\n"
                  << "index   " << duration{t2 - t1}.count() << "s\n";

    } catch (std::exception& ex) {
        std::cerr << "indexing failed: " << ex.what() << '\n' << failure;
    }

    std::cout << success;
}
