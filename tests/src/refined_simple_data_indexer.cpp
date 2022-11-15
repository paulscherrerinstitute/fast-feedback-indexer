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

    try {
        if (argc <= 5)
            throw std::runtime_error("missing arguments <file name> <max number of spots> <max number of output cells> <number of kept candidate vectors> <number of half sphere sample points>");

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

        Eigen::MatrixX3f cells(3 * cpers.max_output_cells, 3); // output cells coordinate container
        Eigen::VectorXf scores(cpers.max_output_cells); // output cell scores container
        fast_feedback::indexer indexer{cpers};          // indexer object

        fast_feedback::memory_pin pin_coords{coords};   // pin input coordinate container
        fast_feedback::memory_pin pin_cells{cells};     // pin output cells coordinate container
        fast_feedback::memory_pin pin_scores(scores);   // pin output cell scores container
        fast_feedback::memory_pin pin_crt{fast_feedback::memory_pin::on(crt)}; // pin runtime config memory

        fast_feedback::input<float> in{&coords(0,0), &coords(0,1), &coords(0,2), 1u, i-3u}; // create indexer input object
        fast_feedback::output<float> out{&cells(0,0), &cells(0,1), &cells(0,2), scores.data(), cpers.max_output_cells}; // create indexer output object

        indexer.index(in, out, crt);                    // run indexer

        {                                               // refine cells
            using namespace Eigen;
            MatrixX3f spots = coords.bottomRows(coords.rows() - 3);
            for (unsigned j=0u; j<out.n_cells; j++) {
                Matrix3f cell = cells.block(3 * j, 0, 3, 3).inverse();
                ArrayX3f temp = spots * cell;                // coordinates in system <cell>
                ArrayX3f miller = round(temp);
                temp -= miller;
                Vector<bool, Dynamic> thresh(temp.cols());
                thresh = temp.abs().rowwise().maxCoeff() < .2f;
                Matrix<bool, Dynamic, 3> sel(thresh.size(), 3);
                sel.colwise() = thresh;
                FullPivHouseholderQR<Eigen::MatrixX3f> qr(sel.select(miller, .0f));
                cell = qr.solve(sel.select(spots, .0f));
                cells.block(3 * j, 0, 3, 3) = cell;
                temp = spots * cell.inverse();
                temp -= miller;
                thresh = temp.abs().rowwise().maxCoeff() < .1f;
                scores(j) = (float)std::reduce(std::begin(thresh), std::end(thresh), 0u, [](unsigned a, unsigned b)->unsigned {return a + b;}) / (float)spots.rows();
            }
        }

        std::cout << "output:\n" << cells << '\n';      // refined output cells
        std::cout << "scores:\n" << scores << '\n';     // scores

    } catch (std::exception& ex) {
        std::cerr << "indexing failed: " << ex.what() << '\n' << failure;
    }

    std::cout << success;
}
