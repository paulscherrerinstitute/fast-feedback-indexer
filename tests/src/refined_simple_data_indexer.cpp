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
#include <numeric>
#include <chrono>
#include <Eigen/Dense>
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
    using clock = std::chrono::high_resolution_clock;
    using duration = std::chrono::duration<double>;

    try {
        if (argc <= 7)
            throw std::runtime_error("missing arguments <file name> <max number of spots> <max number of output cells> <number of kept candidate vectors> <number of half sphere sample points> <lsq spot filter threshold> <score filter threshold>");

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

        float lsq_threshold = .2f;
        {
            std::istringstream iss(argv[6]);
            iss >> lsq_threshold;
            if (! iss)
                throw std::runtime_error("unable to parse fifth argument: threshold for spot selection in least squares fitting");
            std::cout << "lsq_threshold=" << lsq_threshold << '\n';
            if ((lsq_threshold <= .0f) || (lsq_threshold > .5f))
                throw std::runtime_error("lsq_threshold must be in range (0..0.5]");
        }

        float score_threshold = .1f;
        {
            std::istringstream iss(argv[7]);
            iss >> score_threshold;
            if (! iss)
                throw std::runtime_error("unable to parse sixth argument: threshold for spot selection in score calculation");
            std::cout << "score_threshold=" << score_threshold << '\n';
            if ((score_threshold <= .0f) || (score_threshold > .5f))
                throw std::runtime_error("score_threshold must be in range (0..0.5]");
        }

        SimpleData<float, raise> data(argv[1]);         // read simple data file

        Eigen::MatrixX3f coords(data.spots.size() + 3, 3); // coordinate container
        unsigned i=0;
        for (const auto& coord : data.unit_cell) {      // copy cell coordinates
            coords(i, 0) = coord.x;
            coords(i, 1) = coord.y;
            coords(i, 2) = coord.z;
            i++;
        }
        std::cout << "input:\n" << coords.block(0, 0, 3, 3) << '\n';
        for (const auto& coord : data.spots) {          // copy spot coordinates
            coords(i, 0) = coord.x;
            coords(i, 1) = coord.y;
            coords(i, 2) = coord.z;
            i++;            
        }

        auto t0 = clock::now();

        Eigen::MatrixX3f cells(3 * cpers.max_output_cells, 3);  // output cells coordinate container
        Eigen::MatrixX2f scores(cpers.max_output_cells, 2);     // output cell scores container
        fast_feedback::indexer indexer{cpers};          // indexer object

        fast_feedback::memory_pin pin_coords{coords};   // pin input coordinate container
        fast_feedback::memory_pin pin_cells{cells};     // pin output cells coordinate container
        fast_feedback::memory_pin pin_scores(scores);   // pin output cell scores container
        fast_feedback::memory_pin pin_crt{fast_feedback::memory_pin::on(crt)}; // pin runtime config memory

        fast_feedback::input<float> in{&coords(0,0), &coords(0,1), &coords(0,2), 1u, i-3u};                             // create indexer input object
        fast_feedback::output<float> out{&cells(0,0), &cells(0,1), &cells(0,2), scores.data(), cpers.max_output_cells}; // create indexer output object

        auto t1 = clock::now();
        indexer.index(in, out, crt);                    // run indexer
        auto t2 = clock::now();

        {                                               // refine cells
            using namespace Eigen;
            MatrixX3f spots = coords.bottomRows(coords.rows() - 3);
            for (unsigned j=0u; j<out.n_cells; j++) {
                Matrix3f cell = cells.block(3 * j, 0, 3, 3).inverse();
                MatrixX3f temp = spots * cell;          // coordinates in system <cell>
                MatrixX3f miller = round(temp.array());
                temp -= miller;
                Vector<bool, Dynamic> thresh(temp.cols());
                thresh = temp.array().abs().rowwise().maxCoeff() < lsq_threshold;
                Matrix<bool, Dynamic, 3> sel(thresh.size(), 3);
                sel.colwise() = thresh;
                FullPivHouseholderQR<Eigen::MatrixX3f> qr(sel.select(miller, .0f));
                cell = qr.solve(sel.select(spots, .0f));
                cells.block(3 * j, 0, 3, 3) = cell;
                temp = spots * cell.inverse();
                temp -= miller;
                thresh = temp.array().abs().rowwise().maxCoeff() < score_threshold;
                scores(j, 1) = (float)std::reduce(std::begin(thresh), std::end(thresh), 0u, [](unsigned a, unsigned b)->unsigned {return a + b;}) / (float)spots.rows();
                scores(j, 0) /= -(3.f * spots.rows());
            }
        }

        auto t3 = clock::now();

        std::cout << "output:\n";
        for (unsigned j=0u; j<out.n_cells; j++)
            std::cout << cells.block(3 * j, 0, 3, 3) << "\n\n"; // refined output cells
        std::cout << "scores:\n" << scores << '\n';             // scores
        std::cout << "timings:\n"
                  << "prep    " << duration{t1 - t0}.count() << "s\n"
                  << "index   " << duration{t2 - t1}.count() << "s\n"
                  << "refine  " << duration{t3 - t2}.count() << "s\n"
                  << "i+r     " << duration{t3 - t1}.count() << "s\n";

    } catch (std::exception& ex) {
        std::cerr << "indexing failed: " << ex.what() << '\n' << failure;
    }

    std::cout << success;
}
