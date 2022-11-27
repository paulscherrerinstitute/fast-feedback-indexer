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

#ifndef INDEXER_REFINE_H
#define INDEXER_REFINE_H

#include <Eigen/Dense>
#include <Eigen/LU>
#include <exception.h>
#include <numeric>
#include <functional>
#include "indexer.h"

namespace fast_feedback {
    namespace refine {

        // base indexer class for refinement
        // - controlling all indexer data
        // - getter/setter interface
        template<typename float_type=float>
        class indexer {
          protected:
            Eigen::Matrix<float_type, Eigen::Dynamic, 3u> coords;  // for input cells + spots in reciprocal space
            Eigen::Matrix<float_type, Eigen::Dynamic, 3u> cells;   // output cells coordinate container
            Eigen::Vector<float_type, Eigen::Dynamic> scores;      // output cell scores container
            fast_feedback::config_runtime<float_type> crt;
            fast_feedback::indexer<float_type> idx;
            fast_feedback::memory_pin pin_coords;                   // pin input coordinate container
            fast_feedback::memory_pin pin_cells;                    // pin output cells coordinate container
            fast_feedback::memory_pin pin_scores;                   // pin output cell scores container
            fast_feedback::memory_pin pin_crt;                      // pin runtime config memory
          public:
            inline static void check_config (const fast_feedback::config_persistent<float_type>& cp,
                                             const fast_feedback::config_runtime<float_type>& cr)
            {
                if (cp.max_input_cells <= 0u)
                    throw FF_EXCEPTION("no input cells");
                if (cp.max_output_cells <= 0)
                    throw FF_EXCEPTION("no output cells");
                if (cp.max_spots <= 0u)
                    throw FF_EXCEPTION("no spots");
                if (cp.num_candidate_vectors < 1u)
                    throw FF_EXCEPTION("nonpositive number of candidate vectors");
                if (cr.num_sample_points < cp.num_candidate_vectors)
                    throw FF_EXCEPTION("fewer sample points than required candidate vectors");
            }

            inline indexer (const fast_feedback::config_persistent<float_type>& cp,
                            const fast_feedback::config_runtime<float_type>& cr)
                : coords{cp.max_spots + 3u * cp.max_input_cells, 3u},
                  cells{3u * cp.max_output_cells, 3u}, scores{cp.max_output_cells},
                  crt{cr}, idx{cp},
                  pin_coords{coords}, pin_cells{cells}, pin_scores{scores},
                  pin_crt{fast_feedback::memory_pin::on(cr)}
            {
                check_config(cp, cr);
            }

            inline indexer (indexer&&) = default;
            inline indexer& operator= (indexer&&) = default;
            inline virtual ~indexer () = default;

            indexer () = delete;
            indexer (const indexer&) = delete;
            indexer& operator= (const indexer&) = delete;

            inline virtual void index (unsigned n_input_cells, unsigned n_spots)
            {
                fast_feedback::input<float_type> input{&coords(0,0), &coords(0,1), &coords(0,2), n_input_cells, n_spots};
                fast_feedback::output<float_type> output{&cells(0,0), &cells(0,1), &cells(0,2), scores.data(), idx.cpers.max_output_cells};
                idx.index(input, output, crt);
            }

            // spot access: spot i
            inline float_type& spotX (unsigned i=0u) noexcept
            { return coords(3u * idx.cpers.max_input_cells + i, 0u); }

            inline const float_type& spotX (unsigned i) const noexcept
            { return &coords(3u * idx.cpers.max_input_cells + i, 0u); }

            inline float_type& spotY (unsigned i=0u) noexcept
            { return coords(3u * idx.cpers.max_input_cells + i, 1u); }

            inline const float_type& spotY (unsigned i=0u) const noexcept
            { return coords(3u * idx.cpers.max_input_cells + i, 1u); }

            inline float_type& spotZ (unsigned i=0u) noexcept
            { return coords(3u * idx.cpers.max_input_cells + i, 2u); }

            inline const float_type& spotZ (unsigned i=0u) const noexcept
            { return coords(3u * idx.cpers.max_input_cells + i, 2u); }

            inline auto spotM ()
            { return coords.bottomRows(coords.rows() - 3u * idx.cpers.max_input_cells); }

            // input cell access: cell i, vector j
            inline float_type& iCellX (unsigned i=0u, unsigned j=0u) noexcept
            { return coords(3u * i + j, 0u); }

            inline const float_type& iCellX (unsigned i=0u, unsigned j=0u) const noexcept
            { return coords(3u * i + j, 0u); }

            inline float_type& iCellY (unsigned i=0u, unsigned j=0u) noexcept
            { return coords(3u * i + j, 1u); }

            inline const float_type& iCellY (unsigned i=0u, unsigned j=0u) const noexcept
            { return coords(3u * i + j, 1u); }

            inline float_type& iCellZ (unsigned i=0u, unsigned j=0u) noexcept
            { return coords(3u * i + j, 2u); }

            inline const float_type& iCellZ (unsigned i=0u, unsigned j=0u) const noexcept
            { return coords(3u * i + j, 2u); }

            inline auto iCellM ()
            { return coords.topRows(3u * idx.cpers.max_input_cells); }

            // output cell access: cell i, vector j
            inline const float_type& oCellX (unsigned i=0u, unsigned j=0u) const noexcept
            { return cells(3u * i + j, 0u); }

            inline const float_type& oCellY (unsigned i=0u, unsigned j=0u) const noexcept
            { return cells(3u * i + j, 1u); }

            inline const float_type& oCellZ (unsigned i=0u, unsigned j=0u) const noexcept
            { return cells(3u * i + j, 2u); }

            inline auto& oCellM () noexcept
            { return cells; }

            // output cell score access: cell i
            inline const float_type& oScore (unsigned i=0u) const noexcept
            { return scores(i); }

            inline auto& oScoreV () noexcept
            { return scores; }

            // runtime configuration access
            inline void length_threshold (float_type lt)
            {
                if (lt < float_type{.0f})
                    throw FF_EXCEPTION("negative length threshold");
                crt.length_threshold = lt;
            }

            inline float_type length_threshold () const noexcept
            { return crt.length_threshold; }

            inline void num_sample_points (unsigned nsp)
            {
                unsigned tmp = crt.num_sample_points;
                crt.num_sample_points = nsp;
                try {
                    check_config();
                } catch (...) {
                    crt.num_sample_points = tmp;
                    throw;
                }
            }

            inline unsigned num_sample_points () const noexcept
            { return crt.num_sample_points; }

            const config_runtime<float_type>& conf_runtime () const noexcept
            { return crt; }

            // persistent configuration access
            // - to change the persistent config, create another indexer instance
            inline unsigned max_output_cells () const noexcept
            { return idx.cpers.max_output_cells; }

            inline unsigned max_input_cells () const noexcept
            { return idx.cpers.max_input_cells; }

            inline unsigned max_spots () const noexcept
            { return idx.cpers.max_spots; }

            inline unsigned num_candidate_vectors () const noexcept
            { return idx.cpers.num_candidate_vectors; }

            const config_persistent<float_type>& conf_persistent () const noexcept
            { return idx.cpers; }

        }; // indexer

        // least squares refinement indexer extra config
        template <typename float_type=float>
        struct config_lsq final {
            float_type fit_threshold=.1;    // fit unit cells to spots closer than this to approximated Miller indices
            float_type score_threshold=.2;  // calculate score as percentage of spots closer than this to approximated Miller indices
        };

        // least squares refinement indexer
        template <typename float_type=float>
        class indexer_lsq : public indexer<float_type> {
            config_lsq<float_type> clsq;
          public:
            indexer_lsq (const fast_feedback::config_persistent<float_type>& cp,
                         const fast_feedback::config_runtime<float_type>& cr,
                         const config_lsq<float_type>& c)
                : indexer<float_type>{cp, cr}, clsq{c}
            {}

            inline indexer_lsq (indexer_lsq&&) = default;
            inline indexer_lsq& operator= (indexer_lsq&&) = default;
            inline ~indexer_lsq () override = default;

            indexer_lsq () = delete;
            indexer_lsq (const indexer_lsq&) = delete;
            indexer_lsq& operator= (const indexer_lsq&) = delete;

            // refine output
            void refine ()
            {
                using namespace std;
                using namespace Eigen;
                auto& coords = indexer<float_type>::coords;
                auto& cells = indexer<float_type>::cells;
                auto& scores = indexer<float_type>::scores;
                auto& cpers = indexer<float_type>::idx.cpers;
                MatrixX3f spots = coords.bottomRows(coords.rows() - 3u * cpers.max_input_cells);
                for (unsigned j=0u; j<cpers.max_output_cells; j++) {
                    Matrix3f cell = cells.block(3u * j, 0u, 3u, 3u).inverse();
                    MatrixX3f temp = spots * cell;          // coordinates in system <cell>
                    MatrixX3f miller = round(temp.array());
                    temp -= miller;
                    Vector<bool, Dynamic> thresh(temp.cols());
                    thresh = temp.array().abs().rowwise().maxCoeff() < clsq.fit_threshold;
                    Matrix<bool, Dynamic, 3u> sel(thresh.size(), 3u);
                    sel.colwise() = thresh;
                    FullPivHouseholderQR<MatrixX3f> qr(sel.select(miller, .0f));
                    cell = qr.solve(sel.select(spots, .0f));
                    cells.block(3u * j, 0u, 3u, 3u) = cell;
                    temp = spots * cell.inverse();
                    temp -= miller;
                    thresh = temp.array().abs().rowwise().maxCoeff() < clsq.score_threshold;
                    scores(j) = static_cast<float_type>(reduce(begin(thresh), end(thresh), 0u, plus<unsigned>{}))
                                / static_cast<float_type>(spots.rows());
                }
            }

            // refined indexing
            void index (unsigned n_input_cells, unsigned n_spots) override
            {
                indexer<float_type>::index(n_input_cells, n_spots);
                refine();
            }

            // lsq configuration access
            inline void fit_threshold (float_type ft) noexcept
            { clsq.fit_threshold = ft; }
            
            inline float_type fit_threshold () const noexcept
            { return clsq.fit_threshold; }

            inline void score_threshold (float_type st) noexcept
            { clsq.score_threshold = st; }

            inline float_type score_threshold () const noexcept
            { return clsq.score_threshold; }

            const config_lsq<float_type>& conf_lsq () const noexcept
            { return clsq; }

        }; // indexer_lsq
            
    } // namespace refine
} // namespace fast_feedback

#endif // INDEXER_REFINE_H
