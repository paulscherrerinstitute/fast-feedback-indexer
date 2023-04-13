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

//#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <numeric>
#include <functional>
#include <algorithm>
#include <chrono>
#include <ffbidx/exception.h>
#include "ffbidx/indexer.h"
#include "ffbidx/log.h"

namespace fast_feedback {
    namespace refine {
        // Provide convenience classes for the fast feedback indexer
        // - indexer: calls the raw fast_feedback::indexer
        // - indexer_ifss: calls indexer and refined the output cells
        // - indexer_ifse: calls indexer and refines the output cells
        //
        // Like the raw indexer these indexers (idx) provide several call possibilities
        //
        // - synchronous: idx.index(..)
        // - asynchronous: idx.index_start(.., callback, data)
        //                      callback(data) by GPU <-- result is ready
        //                 idx.index_end() <-- will do the refinement on CPU
        //
        // For multithreaded refinement use the base indexer, split the output cells into
        // logical blocks, and let threads call the refine(..) static methods on individual
        // output cell blocks.

        using logger::stanza;

        // Base indexer class for refinement
        // - controlling all indexer data
        // - getter/setter interface
        // The origin is implicitly assumed to be part of the lattice
        template<typename float_type=float>
        class indexer {
          protected:
            fast_feedback::indexer<float_type> idx;                 // raw indexer
            Eigen::MatrixX3<float_type> icells;                     // space for max_input_cells in real space
            Eigen::MatrixX3<float_type> spots;                      // space for max_spots in reciprocal space
            Eigen::MatrixX3<float_type> ocells;                     // max_output_cells coordinate container
            Eigen::VectorX<float_type> scores;                      // output cell scores container
            fast_feedback::input<float_type> input;                 // raw indexer input
            fast_feedback::output<float_type> output;               // raw indexer output
            fast_feedback::memory_pin pin_icells;                   // pin input cells container
            fast_feedback::memory_pin pin_spots;                    // pin spots container
            fast_feedback::memory_pin pin_ocells;                   // pin output cells container
            fast_feedback::memory_pin pin_scores;                   // pin output cell scores container
            fast_feedback::memory_pin pin_crt;                      // pin runtime config memory
            fast_feedback::config_runtime<float_type> crt;          // raw indexer runtime config
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
                if (cr.triml < float_type{0.f})
                    throw FF_EXCEPTION("lower trim value < 0");
                if (cr.triml > cr.trimh)
                    throw FF_EXCEPTION("lower > higher trim value");
                if (cr.trimh > float_type{.5f})
                    throw FF_EXCEPTION("higher trim value > 0.5");
                if (cr.delta <= float_type{.0})
                    throw FF_EXCEPTION("nonpositive delta value");
            }

            // Create base indexer object
            // - allocate (pinned) memory for all input and output data
            // - store configuration
            inline indexer (const fast_feedback::config_persistent<float_type>& cp,
                            const fast_feedback::config_runtime<float_type>& cr)
                : idx{cp},
                  icells{cp.max_input_cells * 3u, 3u}, spots{cp.max_spots, 3u},
                  ocells{3u * cp.max_output_cells, 3u}, scores{cp.max_output_cells},
                  input{{&icells(0,0), &icells(0,1), &icells(0,2)}, {&spots(0,0), &spots(0,1), &spots(0,2)}, 0u, 0u, true, true},
                  output{&ocells(0,0), &ocells(0,1), &ocells(0,2), scores.data(), idx.cpers.max_output_cells},
                  pin_icells{icells}, pin_spots{spots},
                  pin_ocells{ocells}, pin_scores{scores},
                  pin_crt{fast_feedback::memory_pin::on(crt)},
                  crt{cr}
            {
                check_config(cp, cr);
            }

            inline indexer (indexer&&) = default;
            inline indexer& operator= (indexer&&) = default;
            inline virtual ~indexer () = default;

            indexer () = delete;
            indexer (const indexer&) = delete;
            indexer& operator= (const indexer&) = delete;

            // Asynchronouly launch indexing operation on GPU
            // - n_input_cells must be less than max_input_cells
            // - n_spots must be less than max_spots
            // - if callback is given, it will be called with data as the argument as soon as  index_end can be called
            inline void index_start (unsigned n_input_cells, unsigned n_spots, void(*callback)(void*)=nullptr, void* data=nullptr)
            {
                input.n_cells = n_input_cells;
                input.n_spots = n_spots;
                idx.index_start(input, output, crt, callback, data);
            }

            // Finish indexing
            // - index_start must have been called before this
            inline virtual void index_end ()
            {
                idx.index_end(output);
            }

            // Synchronous indexing
            // - n_input_cells must be less than max_input_cells
            // - n_spots must be less than max_spots
            inline void index (unsigned n_input_cells, unsigned n_spots)
            {
                index_start(n_input_cells, n_spots);
                index_end();
            }

            // Reciprocal space spot access: spot i
            inline float_type& spotX (unsigned i=0u) noexcept
            { return spots(i, 0u); }

            inline const float_type& spotX (unsigned i=0u) const noexcept
            { return &spots(i, 0u); }

            inline float_type& spotY (unsigned i=0u) noexcept
            { return spots(i, 1u); }

            inline const float_type& spotY (unsigned i=0u) const noexcept
            { return spots(i, 1u); }

            inline float_type& spotZ (unsigned i=0u) noexcept
            { return spots(i, 2u); }

            inline const float_type& spotZ (unsigned i=0u) const noexcept
            { return spots(i, 2u); }

            // Coords area designated for reciprocal space spots, fill top down
            inline auto& spotM ()
            { return spots; }

            // Used coords area designated for reciprocal space spots
            inline const auto Spots ()
            {
                return spots.topRows(input.n_spots);
            }

            // Real space input cell access: cell i, vector j
            inline float_type& iCellX (unsigned i=0u, unsigned j=0u) noexcept
            { return icells(3u * i + j, 0u); }

            inline const float_type& iCellX (unsigned i=0u, unsigned j=0u) const noexcept
            { return icells(3u * i + j, 0u); }

            inline float_type& iCellY (unsigned i=0u, unsigned j=0u) noexcept
            { return icells(3u * i + j, 1u); }

            inline const float_type& iCellY (unsigned i=0u, unsigned j=0u) const noexcept
            { return icells(3u * i + j, 1u); }

            inline float_type& iCellZ (unsigned i=0u, unsigned j=0u) noexcept
            { return icells(3u * i + j, 2u); }

            inline const float_type& iCellZ (unsigned i=0u, unsigned j=0u) const noexcept
            { return icells(3u * i + j, 2u); }

            // Coords area for real space input cell i
            inline auto iCell (unsigned i=0u) noexcept
            { return icells.block(3u * i, 0u, 3u, 3u); }

            inline const auto iCell (unsigned i=0u) const noexcept
            { return icells.block(3u * i, 0u, 3u, 3u); }

            // Coords area designated for real space input cells, fill cellwise bottom up
            inline auto iCellM () noexcept
            { return icells.topRows(3u * idx.cpers.max_input_cells); }

            // Input size access
            inline unsigned n_input_cells () noexcept
            { return input.n_cells; }

            inline unsigned n_spots () noexcept
            { return input.n_spots; }

            // Real space output cell access: cell i, vector j
            inline float_type& oCellX (unsigned i=0u, unsigned j=0u) noexcept
            { return ocells(3u * i + j, 0u); }

            inline const float_type& oCellX (unsigned i=0u, unsigned j=0u) const noexcept
            { return ocells(3u * i + j, 0u); }

            inline float_type& oCellY (unsigned i=0u, unsigned j=0u) noexcept
            { return ocells(3u * i + j, 1u); }

            inline const float_type& oCellY (unsigned i=0u, unsigned j=0u) const noexcept
            { return ocells(3u * i + j, 1u); }

            inline float_type& oCellZ (unsigned i=0u, unsigned j=0u) noexcept
            { return ocells(3u * i + j, 2u); }

            inline const float_type& oCellZ (unsigned i=0u, unsigned j=0u) const noexcept
            { return ocells(3u * i + j, 2u); }

            // Real space output cell i
            inline auto oCell (unsigned i) noexcept
            { return ocells.block(3u * i, 0u, 3u, 3u); }

            inline const auto oCell (unsigned i) const noexcept
            { return ocells.block(3u * i, 0u, 3u, 3u); }

            // All output cells
            inline auto& oCellM () noexcept
            { return ocells; }

            // Output cell score access: cell i
            inline float_type& oScore (unsigned i=0u) noexcept
            { return scores(i); }

            inline const float_type& oScore (unsigned i=0u) const noexcept
            { return scores(i); }

            // All output scores
            inline auto& oScoreV () noexcept
            { return scores; }

            // Dissect the base indexer score into two parts:
            // - main score: number of spots within a distance of trimh to closest lattice point
            // - sub score: exp2(sum[spots](log2(trim[triml..trimh](dist2int(dot(v,spot))) + delta)) / #spots) - delta
            inline static std::pair<float_type, float_type> score_parts (float_type score)
            {
                float_type nsp = -std::floor(score);
                float_type s = score + nsp;
                return std::make_pair(nsp, s);
            }

            // Output size access
            inline unsigned n_output_cells ()
            { return output.n_cells; }

            // Runtime configuration access
            inline void length_threshold (float_type lt)
            {
                if (lt < float_type{.0f})
                    throw FF_EXCEPTION("negative length threshold");
                crt.length_threshold = lt;
            }

            inline float_type length_threshold () const noexcept
            { return crt.length_threshold; }

            inline void triml (float_type tl)
            {
                if (tl < float_type{0.f})
                    throw FF_EXCEPTION("lower trim value < 0");
                if (tl > crt.trimh)
                    throw FF_EXCEPTION("lower > higher trim value");
                crt.triml = tl;
            }

            inline float_type triml () const noexcept
            {
                return crt.triml;
            }

            inline void trimh (float_type th)
            {
                if (crt.triml > th)
                    throw FF_EXCEPTION("lower > higher trim value");
                if (th > float_type{.5f})
                    throw FF_EXCEPTION("higher trim value > 0.5");
                crt.trimh = th;
            }

            inline float_type trimh () const noexcept
            {
                return crt.trimh;
            }

            inline void delta (float_type d)
            {
                if (d <= float_type{.0})
                    throw FF_EXCEPTION("nonnegative delta value");
            }

            inline float_type delta () const noexcept
            {
                return crt.delta;
            }

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

            inline const config_runtime<float_type>& conf_runtime () const noexcept
            { return crt; }

            // Persistent configuration access
            // - to change the persistent config, create another indexer instance
            inline unsigned max_output_cells () const noexcept
            { return idx.cpers.max_output_cells; }

            inline unsigned max_input_cells () const noexcept
            { return idx.cpers.max_input_cells; }

            inline unsigned max_spots () const noexcept
            { return idx.cpers.max_spots; }

            inline unsigned num_candidate_vectors () const noexcept
            { return idx.cpers.num_candidate_vectors; }

            inline const config_persistent<float_type>& conf_persistent () const noexcept
            { return idx.cpers; }
        }; // indexer

        // Iterative fit to selected spots refinement indexer extra config
        template <typename float_type=float>
        struct config_ifss final {
            float_type threshold_contraction=.8;    // contract error threshold by this value in every iteration
            unsigned min_spots=6;                   // minimum number of spots to fit against
            unsigned max_iter=15;                   // max number of iterations
        };

        // Iterative fit to selected spots refinement indexer
        template <typename float_type=float>
        class indexer_ifss : public indexer<float_type> {
            config_ifss<float_type> cifss;
          public:
            inline static void check_config (const config_ifss<float_type>& c)
            {
                if (c.threshold_contraction <= float_type{.0})
                    throw FF_EXCEPTION("nonpositive threshold contraction");
                if (c.threshold_contraction >= float_type{1.})
                    throw FF_EXCEPTION("threshold contraction >= 1");
                if (c.min_spots <= 3)
                    throw FF_EXCEPTION("min spots <= 3");
            }

            inline indexer_ifss (const fast_feedback::config_persistent<float_type>& cp,
                                const fast_feedback::config_runtime<float_type>& cr,
                                const config_ifss<float_type>& c)
                : indexer<float_type>{cp, cr}, cifss{c}
            {
                check_config(c);
            }

            inline indexer_ifss (indexer_ifss&&) = default;
            inline indexer_ifss& operator= (indexer_ifss&&) = default;
            inline ~indexer_ifss () override = default;

            indexer_ifss () = delete;
            indexer_ifss (const indexer_ifss&) = delete;
            indexer_ifss& operator= (const indexer_ifss&) = delete;

            // Refine cells
            //
            // This call splits cells into nblocks cell blocks to allow multithreaded cell refinement.
            // All threads must use a common nblocks parameter and each thread an individual block parameter.
            //
            // input:
            // - spots      spot reciprocal coordinates matrix
            // - cells      output cells real space coordinates matrix like the one in the base indexer
            // - scores     output cell scores matrix with scores coming from the base indexer
            // - cpers      persistent config for the matrices
            // - cifss      ifss config
            // - block      which of the N cell blocks
            // - nblocks    use N cell blocks
            // output:
            // - cells      the refined cells
            // - scores     refined cell scores: largest distance of the min_spots closest to their approximated lattice points
            template<typename MatX3, typename VecX>
            inline static void refine (const Eigen::Ref<Eigen::MatrixX3<float_type>>& spots,
                                       Eigen::DenseBase<MatX3>& cells,
                                       Eigen::DenseBase<VecX>& scores,
                                       const config_ifss<float_type>& cifss,
                                       unsigned block=0, unsigned nblocks=1)
            {
                using namespace Eigen;
                using Mx3 = MatrixX3<float_type>;
                using M3 = Matrix3<float_type>;
                const unsigned nspots = spots.rows();
                const unsigned ncells = scores.rows();
                VectorX<bool> below{nspots};
                MatrixX3<bool> sel{nspots, 3u};
                Mx3 resid{nspots, 3u};
                Mx3 miller{nspots, 3u};
                M3 cell;
                const unsigned blocksize = (ncells + nblocks - 1u) / nblocks;
                const unsigned startcell = block * blocksize;
                const unsigned endcell = std::min(startcell + blocksize, ncells);
                for (unsigned j=startcell; j<endcell; j++) {
                    cell = cells.block(3u * j, 0u, 3u, 3u).transpose();  // cell: col vectors
                    float_type threshold = indexer<float_type>::score_parts(scores[j]).second;
                    for (unsigned niter=0; niter<cifss.max_iter; niter++) {
                        resid = spots * cell;   // coordinates in system <cell>
                        miller = round(resid.array());
                        resid -= miller;
                        below = (resid.rowwise().norm().array() < threshold);
                        if (below.count() < cifss.min_spots)
                            break;
                        threshold *= cifss.threshold_contraction;
                        sel.colwise() = below;
                        HouseholderQR<Mx3> qr{sel.select(spots, .0f)};
                        cell = qr.solve(sel.select(miller, .0f));
                    }
                    {
                        ArrayX<float_type> dist = resid.rowwise().norm();
                        const auto front = std::begin(dist);
                        auto back = std::end(dist);
                        const std::greater<float> greater{};
                        std::make_heap(front, back, greater);
                        for (unsigned i=0u; i<=cifss.min_spots; i++)
                            std::pop_heap(front, back, greater), --back;
                        scores(j) = *back;
                    }
                    cells.block(3u * j, 0u, 3u, 3u) = cell.transpose();
                }
            }

            // Refined result
            inline void index_end () override
            {
                indexer<float_type>::index_end();
                refine(this->Spots(), this->ocells, this->scores, cifss);
            }

            // ifss configuration access
            inline void threshold_contraction (float_type tc)
            {
                if (tc <= float_type{.0})
                    throw FF_EXCEPTION("nonpositive threshold contraction");
                if (tc >= float_type{1.})
                    throw FF_EXCEPTION("threshold contraction >= 1");
                cifss.threshold_contraction = tc;
            }
            
            inline float_type threshold_contraction () const noexcept
            { return cifss.threshold_contraction; }

            inline void min_spots (unsigned ms)
            {
                if (ms <= 3)
                    throw FF_EXCEPTION("min spots <= 3");
                cifss.min_spots = ms;
            }

            inline unsigned min_spots () const noexcept
            { return cifss.min_spots; }

            inline void max_iter (unsigned n) noexcept
            { cifss.max_iter = n; }

            inline unsigned max_iter () const noexcept
            { return cifss.max_iter; }

            inline const config_ifss<float_type>& conf_ifss () const noexcept
            { return cifss; }

        }; // indexer_ifss
            
        // Iterative fit to selected errors refinement indexer extra config
        template <typename float_type=float>
        struct config_ifse final {
            float_type threshold_contraction=.8;    // contract error threshold by this value in every iteration
            unsigned min_spots=6;                   // minimum number of spots to fit against
            unsigned max_iter=15;                   // max number of iterations
        };

        // Iterative fit to selected errors refinement indexer
        template <typename float_type=float>
        class indexer_ifse : public indexer<float_type> {
            config_ifse<float_type> cifse;
          public:
            inline static void check_config (const config_ifse<float_type>& c)
            {
                if (c.threshold_contraction <= float_type{.0})
                    throw FF_EXCEPTION("nonpositive contraction speed");
                if (c.min_spots <= 3)
                    throw FF_EXCEPTION("min spots <= 3");
            }

            inline indexer_ifse (const fast_feedback::config_persistent<float_type>& cp,
                          const fast_feedback::config_runtime<float_type>& cr,
                          const config_ifse<float_type>& c)
                : indexer<float_type>{cp, cr}, cifse{c}
            {}

            inline indexer_ifse (indexer_ifse&&) = default;
            inline indexer_ifse& operator= (indexer_ifse&&) = default;
            inline ~indexer_ifse () override = default;

            indexer_ifse () = delete;
            indexer_ifse (const indexer_ifse&) = delete;
            indexer_ifse& operator= (const indexer_ifse&) = delete;

            // Refine cells
            //
            // This call splits cells into nblocks cell blocks to allow multithreaded cell refinement.
            // All threads must use a common nblocks parameter and each thread an individual block parameter.
            //
            // input:
            // - spots      spot coordinate matrix
            // - cells      output cells matrix like the one in the base indexer
            // - scores     output cell scores matrix with scores coming from the base indexer
            // - cpers      persistent config for the matrices
            // - cifse      ifse config
            // - block      which of the N cell blocks
            // - nblocks    use N cell blocks
            // output:
            // - cells      the refined cells
            // - scores     refined cell scores: largest distance of the min_spots closest to their approximated lattice points
            template<typename MatX3, typename VecX>
            inline static void refine (const Eigen::Ref<Eigen::MatrixX3<float_type>>& spots,
                                       Eigen::DenseBase<MatX3>& cells,
                                       Eigen::DenseBase<VecX>& scores,
                                       const config_ifse<float_type>& cifse,
                                       unsigned block=0, unsigned nblocks=1)
            {
                using namespace Eigen;
                using Mx3 = MatrixX3<float_type>;
                using M3 = Matrix3<float_type>;
                const unsigned nspots = spots.rows();
                const unsigned ncells = scores.rows();
                VectorX<bool> below{nspots};
                MatrixX3<bool> sel{nspots, 3u};
                Mx3 resid{nspots, 3u};
                Mx3 miller{nspots, 3u};
                M3 cell;
                const unsigned blocksize = (ncells + nblocks - 1u) / nblocks;
                const unsigned startcell = block * blocksize;
                const unsigned endcell = std::min(startcell + blocksize, ncells);
                for (unsigned j=startcell; j<endcell; j++) {
                    cell = cells.block(3u * j, 0u, 3u, 3u).transpose();  // cell: col vectors
                    float_type threshold = indexer<float_type>::score_parts(scores[j]).second;
                    for (unsigned niter=0; niter<cifse.max_iter; niter++) {
                        resid = spots * cell;   // coordinates in system <cell>
                        miller = round(resid.array());
                        resid -= miller;
                        below = (resid.rowwise().norm().array() < threshold);
                        if (below.count() < cifse.min_spots)
                            break;
                        threshold *= cifse.threshold_contraction;
                        sel.colwise() = below;
                        HouseholderQR<Mx3> qr{sel.select(spots, .0f)};
                        cell -= qr.solve(sel.select(resid, .0f));
                    }
                    {
                        ArrayX<float_type> dist = resid.rowwise().norm();
                        const auto front = std::begin(dist);
                        auto back = std::end(dist);
                        const std::greater<float> greater{};
                        std::make_heap(front, back, greater);
                        for (unsigned i=0u; i<=cifse.min_spots; i++)
                            std::pop_heap(front, back, greater), --back;
                        scores(j) = *back;
                    }
                    cells.block(3u * j, 0u, 3u, 3u) = cell.transpose();
                }
            }

            // Refined result
            inline void index_end () override
            {
                indexer<float_type>::index_end();
                refine(this->Spots(), this->ocells, this->scores, cifse);
            }

            // ifse configuration access
            inline void threshold_contraction (float_type tc)
            {
                if (tc <= float_type{.0})
                    throw FF_EXCEPTION("nonpositive threshold contraction");
                if (tc >= float_type{1.})
                    throw FF_EXCEPTION("threshold contraction >= 1");
                cifse.threshold_contraction = tc;
            }
            
            inline float_type threshold_contraction () const noexcept
            { return cifse.threshold_contraction; }

            inline void min_spots (unsigned ms)
            {
                if (ms <= 3)
                    throw FF_EXCEPTION("min spots <= 3");
                cifse.min_spots = ms;
            }

            inline unsigned min_spots () const noexcept
            { return cifse.min_spots; }

            inline void max_iter (unsigned n) noexcept
            { cifse.max_iter = n; }

            inline unsigned max_iter () const noexcept
            { return cifse.max_iter; }

            inline const config_ifse<float_type>& conf_ifse () const noexcept
            { return cifse; }

        }; // indexer_ifse

        // Return index for the best cell
        template <typename VecX>
        inline unsigned best_cell (const Eigen::DenseBase<VecX>& scores)
        {
            auto it = std::min_element(std::cbegin(scores), std::cend(scores));
            return (unsigned)(it - std::cbegin(scores));
        }

        // Check if a cell looks like a viable unit cell for the spots
        // - cell       cell in real space
        // - spots      spots in reciprocal space
        // - threshold  radius around approximated miller indices
        // - min_spots  minimum number of spots within threshold
        template <typename Mat3, typename MatX3, typename float_type=typename Mat3::Scalar>
        inline bool is_viable_cell (const Eigen::MatrixBase<Mat3>& cell,
                                    const Eigen::MatrixBase<MatX3>& spots,
                                    float_type threshold=.02f, unsigned min_spots=9u)
        {
            using M3x = Eigen::MatrixX3<float_type>;
            M3x resid = spots * cell.transpose();
            const M3x miller = round(resid.array());
            resid -= miller;
            return (resid.rowwise().norm().array() < threshold).count() >= min_spots;
        }

        // Return indices of cells representing crystalls
        // Cell is considered a new crystall, if it differs by more than good n_spots
        // to other crystalls
        template <typename CellMat, typename SpotMat, typename ScoreVec, typename float_type=typename CellMat::Scalar>
        inline std::vector<unsigned> compute_crystalls (const Eigen::MatrixBase<CellMat>& cells,
                                                        const Eigen::MatrixBase<SpotMat>& spots,
                                                        const Eigen::DenseBase<ScoreVec>& scores,
                                                        float_type threshold=.02f, unsigned min_spots=9u)
        {
            using namespace Eigen;
            using Mx3 = MatrixX3<float_type>;
            using Vx = VectorX<bool>;

            auto spots_covered = [&cells, &spots, threshold](unsigned i) -> Vx {
                Mx3 resid = spots * cells.block(3u * i, 0u, 3u, 3u).transpose();
                const Mx3 miller = round(resid.array());
                resid -= miller;
                return (resid.rowwise().norm().array() < threshold);
            };

            unsigned n_spots = spots.rows();
            std::vector<unsigned> crystalls;
            std::vector<Vx> covered;
            unsigned n_cells = cells.rows() / 3;
            for (unsigned int i=0u; i<n_cells; i++) {
                Vx cover = spots_covered(i);
                unsigned cnt = cover.count();
                if (cnt < min_spots)
                    continue;
                for (unsigned k=0u; k<crystalls.size(); k++) {
                    const unsigned j = crystalls[k];
                    Vx common = cover.array() * covered[k].array();
                    unsigned cocnt = common.count();
                    if (cnt - cocnt < n_spots) {
                        if (scores[i] < scores[j]) {
                            crystalls[k] = i;
                            covered[k] = std::move(cover);
                        }
                        goto skip;
                    }
                }
                crystalls.push_back(i);
                covered.emplace_back(std::move(cover));
              skip: ;
            }
            return crystalls;
        }

        // Make a lattice basis right handed
        template <typename CellMat>
        inline void make_right_handed (Eigen::MatrixBase<CellMat>& cell)
        {
            if (cell.determinant() < .0f)
                cell = -cell;
        }

    } // namespace refine
} // namespace fast_feedback

#endif // INDEXER_REFINE_H
