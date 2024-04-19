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
#include <numeric>
#include <algorithm>
#include "ffbidx/exception.h"
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
                if (cr.num_halfsphere_points < cp.num_candidate_vectors)
                    throw FF_EXCEPTION("fewer half sphere sample points than required candidate vectors");
                if ((cr.num_angle_points > 0u) && (cr.num_angle_points < cp.max_output_cells))
                    throw FF_EXCEPTION("fewer angle sample points than required candidate cells");
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
                if (n_input_cells == 0u) {
                    input.new_cells = false;
                } else {
                    input.n_cells = n_input_cells;
                    input.new_cells = true;
                }
                if (n_spots == 0u) {
                    input.new_spots = false;
                } else {
                    input.n_spots = n_spots;
                    input.new_spots = true;
                }
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
            inline auto Spots ()
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

            inline auto iCell (unsigned i=0u) const noexcept
            { return icells.block(3u * i, 0u, 3u, 3u); }

            // Coords area designated for real space input cells, fill cellwise bottom up
            inline auto& iCellM () noexcept
            { return icells; }

            // Input size access
            inline unsigned n_input_cells () const noexcept
            { return input.n_cells; }

            inline unsigned n_spots () const noexcept
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

            inline auto oCell (unsigned i) const noexcept
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
            inline static std::pair<float_type, float_type> score_parts (float_type score) noexcept
            {
                float_type nsp = -std::floor(score);
                float_type s = score + nsp;
                return std::make_pair(nsp-1, s);
            }

            // Output size access
            inline unsigned n_output_cells () const noexcept
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
                crt.delta = d;
            }

            inline float_type delta () const noexcept
            {
                return crt.delta;
            }

            inline void dist1 (float_type d) noexcept
            {
                crt.dist1 = d;
            }

            inline float_type dist1 () const noexcept
            {
                return crt.dist1;
            }

            inline void dist3 (float_type d) noexcept
            {
                crt.dist3 = d;
            }

            inline float_type dist3 () const noexcept
            {
                return crt.dist3;
            }

            inline void num_halfsphere_points (unsigned nhsp)
            {
                unsigned tmp = crt.num_halfsphere_points;
                crt.num_halfsphere_points = nhsp;
                try {
                    check_config();
                } catch (...) {
                    crt.num_halfsphere_points = tmp;
                    throw;
                }
            }

            inline unsigned num_halfsphere_points () const noexcept
            { return crt.num_halfsphere_points; }

            inline void num_angle_points (unsigned nap)
            {
                unsigned tmp = crt.num_angle_points;
                crt.num_angle_points = nap;
                try {
                    check_config();
                } catch (...) {
                    crt.num_angle_points = tmp;
                    throw;
                }
            }

            inline unsigned num_angle_points () const noexcept
            { return crt.num_angle_points; }

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

            inline bool redundant_computations () const noexcept
            { return idx.cpers.redundant_computations; }

            inline const config_persistent<float_type>& conf_persistent () const noexcept
            { return idx.cpers; }
        }; // indexer

        // Iterative fit to selected spots refinement indexer extra config
        template <typename float_type=float>
        struct config_ifss final {
            float_type threshold_contraction=.8;    // contract error threshold by this value in every iteration
            float_type max_distance=.01;            // max distance to integer for inliers
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
                    resid = spots * cell;       // coordinates in system <cell>
                    miller = round(resid.array());
                    resid -= miller;
                    if (threshold > cifss.max_distance) {
                        below = (resid.rowwise().norm().array() < threshold);
                        if (below.count() < cifss.min_spots)
                            goto calc_score;
                        threshold *= cifss.threshold_contraction;
                        sel.colwise() = below;
                        HouseholderQR<Mx3> qr{sel.select(spots, .0f)};
                        cell = qr.solve(sel.select(miller, .0f));
                    }
                    for (unsigned niter=1; niter<cifss.max_iter && threshold>cifss.max_distance; niter++) {
                        resid = (spots * cell) - miller;
                        below = (resid.rowwise().norm().array() < threshold);
                        if (below.count() < cifss.min_spots)
                            break;
                        threshold *= cifss.threshold_contraction;
                        sel.colwise() = below;
                        HouseholderQR<Mx3> qr{sel.select(spots, .0f)};
                        cell = qr.solve(sel.select(miller, .0f));
                    }
                    calc_score: {
                        ArrayX<float_type> dist = resid.rowwise().norm();
                        if (dist.size() < cifss.min_spots) {
                            scores(j) = float_type{1.};
                        } else {
                            auto nth = std::begin(dist) + (cifss.min_spots - 1);
                            std::nth_element(std::begin(dist), nth, std::end(dist));
                            scores(j) = *nth;
                        }
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

            inline void max_distance (float_type d) noexcept
            { cifss.max_distance = d; }

            inline float_type max_distance () const noexcept
            { return cifss.max_distance; }
            
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
            float_type max_distance=.01;            // max distance to integer for inliers
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
                    resid = spots * cell;   // coordinates in system <cell>
                    miller = round(resid.array());
                    resid -= miller;
                    float_type threshold = indexer<float_type>::score_parts(scores[j]).second;
                    for (unsigned niter=0; niter<cifse.max_iter && threshold>cifse.max_distance; niter++) {
                        below = (resid.rowwise().norm().array() < threshold);
                        if (below.count() < cifse.min_spots)
                            break;
                        threshold *= cifse.threshold_contraction;
                        sel.colwise() = below;
                        HouseholderQR<Mx3> qr{sel.select(spots, .0f)};
                        cell -= qr.solve(sel.select(resid, .0f));
                        resid = spots * cell;   // coordinates in system <cell>
                        miller = round(resid.array());
                        resid -= miller;
                    }
                    {
                        ArrayX<float_type> dist = resid.rowwise().norm();
                        if (dist.size() < cifse.min_spots) {
                            scores(j) = float_type{1.};
                        } else {
                            auto nth = std::begin(dist) + (cifse.min_spots - 1);
                            std::nth_element(std::begin(dist), nth, std::end(dist));
                            scores(j) = *nth;
                        }
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

            inline float_type max_distance () const noexcept
            { return cifse.max_distance; }
            
            inline void max_distance (float_type d) noexcept
            { cifse.max_distance = d; }

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

        // Iterative fit to selected spots reciprocal refinement indexer extra config
        template <typename float_type=float>
        struct config_ifssr final {
            float_type threshold_contraction=.8;    // contract error threshold by this value in every iteration
            float_type max_distance=.00075;         // max distance to reciprocal spots for inliers
            unsigned min_spots=8;                   // minimum number of spots to fit against
            unsigned max_iter=32;                   // max number of iterations
        };

        // Iterative fit to selected spots refinement indexer
        template <typename float_type=float>
        class indexer_ifssr : public indexer<float_type> {
            config_ifssr<float_type> cifssr;
          public:
            inline static void check_config (const config_ifssr<float_type>& c)
            {
                if (c.threshold_contraction <= float_type{.0})
                    throw FF_EXCEPTION("nonpositive threshold contraction");
                if (c.threshold_contraction >= float_type{1.})
                    throw FF_EXCEPTION("threshold contraction >= 1");
                if (c.min_spots <= 3)
                    throw FF_EXCEPTION("min spots <= 3");
            }

            inline indexer_ifssr (const fast_feedback::config_persistent<float_type>& cp,
                                  const fast_feedback::config_runtime<float_type>& cr,
                                  const config_ifssr<float_type>& c)
                : indexer<float_type>{cp, cr}, cifssr{c}
            {
                check_config(c);
            }

            inline indexer_ifssr (indexer_ifssr&&) = default;
            inline indexer_ifssr& operator= (indexer_ifssr&&) = default;
            inline ~indexer_ifssr () override = default;

            indexer_ifssr () = delete;
            indexer_ifssr (const indexer_ifssr&) = delete;
            indexer_ifssr& operator= (const indexer_ifssr&) = delete;

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
            // - cifssr     ifssr config
            // - block      which of the N cell blocks
            // - nblocks    use N cell blocks
            // output:
            // - cells      the refined cells
            // - scores     refined cell scores: largest distance of the min_spots closest to their approximated lattice points
            template<typename MatX3, typename VecX>
            inline static void refine (const Eigen::Ref<Eigen::MatrixX3<float_type>>& spots,
                                       Eigen::DenseBase<MatX3>& cells,
                                       Eigen::DenseBase<VecX>& scores,
                                       const config_ifssr<float_type>& cifssr,
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
                    if (nspots < cifssr.min_spots) {
                        scores(j) = float_type{1.};
                        continue;
                    }
                    cell = cells.block(3u * j, 0u, 3u, 3u).transpose(); // cell: col vectors
                    float_type threshold = indexer<float_type>::score_parts(scores[j]).second;
                    for (unsigned niter=1; niter<cifssr.max_iter && threshold>cifssr.max_distance; niter++) {
                        miller = round((spots * cell).array());
                        resid = miller * cell.inverse();    // reciprocal spots induced by <cell>
                        resid -= spots;                     // distance between induced and given spots
                        below = (resid.rowwise().norm().array() < threshold);
                        if (below.count() < cifssr.min_spots)
                            break;
                        threshold *= cifssr.threshold_contraction;
                        sel.colwise() = below;
                        HouseholderQR<Mx3> qr{sel.select(spots, .0f)};
                        cell = qr.solve(sel.select(miller, .0f));
                    }
                    {   // calc score
                        ArrayX<float_type> dist = resid.rowwise().norm();
                        auto nth = std::begin(dist) + (cifssr.min_spots - 1);
                        std::nth_element(std::begin(dist), nth, std::end(dist));
                        scores(j) = *nth;
                    }
                    cells.block(3u * j, 0u, 3u, 3u) = cell.transpose();
                }
            }

            // Refined result
            inline void index_end () override
            {
                indexer<float_type>::index_end();
                refine(this->Spots(), this->ocells, this->scores, cifssr);
            }

            // ifss configuration access
            inline void threshold_contraction (float_type tc)
            {
                if (tc <= float_type{.0})
                    throw FF_EXCEPTION("nonpositive threshold contraction");
                if (tc >= float_type{1.})
                    throw FF_EXCEPTION("threshold contraction >= 1");
                cifssr.threshold_contraction = tc;
            }

            inline float_type threshold_contraction () const noexcept
            { return cifssr.threshold_contraction; }

            inline void min_spots (unsigned ms)
            {
                if (ms <= 3)
                    throw FF_EXCEPTION("min spots <= 3");
                cifssr.min_spots = ms;
            }

            inline unsigned min_spots () const noexcept
            { return cifssr.min_spots; }

            inline void max_distance (float_type d) noexcept
            { cifssr.max_distance = d; }

            inline float_type max_distance () const noexcept
            { return cifssr.max_distance; }

            inline void max_iter (unsigned n) noexcept
            { cifssr.max_iter = n; }

            inline unsigned max_iter () const noexcept
            { return cifssr.max_iter; }

            inline const config_ifssr<float_type>& conf_ifssr () const noexcept
            { return cifssr; }
        }; // indexer_ifssr

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

        // Return indices of cells representing crystals
        // Cell is considered a new crystal, if it differs by more than good n_spots
        // to other crystalls
        template <typename CellMat, typename SpotMat, typename VecX, typename float_type=typename CellMat::Scalar>
        inline std::vector<unsigned> select_crystals (const Eigen::MatrixBase<CellMat>& cells,
                                                      const Eigen::MatrixBase<SpotMat>& spots,
                                                      const Eigen::DenseBase<VecX>& scores,
                                                      float_type threshold=.02f, unsigned min_spots=9u)
        {
            using namespace Eigen;
            using Mx3 = MatrixX3<float_type>;
            using Vx = VectorX<bool>;

            std::vector<unsigned> crystals;
            const unsigned n_spots = spots.rows();
            const unsigned n_cells = scores.size();
            std::vector<unsigned> sorted(n_cells);

            std::iota(std::begin(sorted), std::end(sorted), 0u);
            std::sort(std::begin(sorted), std::end(sorted), [&scores](const unsigned& a, const unsigned&b) {
                return scores[a] < scores[b];
            });

            auto spots_covered = [&cells, &spots, threshold](unsigned i) -> Vx {
                Mx3 resid = spots * cells.block(3u * i, 0u, 3u, 3u).transpose();
                const Mx3 miller = round(resid.array());
                resid -= miller;
                return (resid.rowwise().norm().array() < threshold);
            };

            Vx allcover = spots_covered(sorted[0]);
            if (allcover.count() < min_spots)
                return crystals;
            crystals.push_back(sorted[0]);

            auto new_spots_filt = [&allcover, n_spots](auto& cellcover) -> unsigned {
                unsigned cnt = 0u;
                for (unsigned i=0; i<n_spots; i++) {
                    if (allcover[i])
                        cellcover[i] = false;   // set spots already covered to false in cellcover
                    else if (cellcover[i])
                        cnt++;
                }
                return cnt; // number of spots covered by cellcover, but not by allcover
            };

            for (unsigned i=1u; i<n_cells; i++) {
                Vx cellcover = spots_covered(sorted[i]);
                const unsigned cnt = new_spots_filt(cellcover);    // modifies cellcover
                if (cnt < min_spots)
                    continue;
                allcover += cellcover;
                crystals.push_back(sorted[i]);
            }
            return crystals;
        }

        // Make a lattice basis right handed
        template <typename CellMat>
        inline void make_right_handed (Eigen::MatrixBase<CellMat>& cell)
        {
            if (cell.determinant() < .0f)
                cell = -cell;
        }

        // Compute a cell similarity score based on cell vector length
        // It can be used for penalisation of cell scores for output cells dissimilar to an input cell
        // max(0, m-t)**2, with m the maximum of the relative cell vector length differences plus the relative determinant difference
        template <typename CellMatA, typename CellMatB, typename float_type=typename CellMatA::Scalar>
        inline float_type cell_similarity (const Eigen::MatrixBase<CellMatA>& cellA, const Eigen::MatrixBase<CellMatB>& cellB, float_type threshold=.02f)
        {
            Eigen::Vector3<typename CellMatA::Scalar> vlen_a = cellA.rowwise().norm();
            Eigen::Vector3<typename CellMatB::Scalar> vlen_b = cellB.rowwise().norm();
            std::sort(std::begin(vlen_a), std::end(vlen_a));
            std::sort(std::begin(vlen_b), std::end(vlen_b));
            const float_type detA = cellA.determinant();
            const float_type dD = std::abs((cellB.determinant() - detA) / detA);
            float_type dL = .0f;
            for (unsigned i=0; i<3u; i++)
                dL = std::max(dL, std::abs(vlen_a[i] - vlen_b[i]) / vlen_a[i]);
            const float_type score = dD + dL;
            return std::max(float_type{.0f}, score * score - threshold);
        }

    } // namespace refine
} // namespace fast_feedback

#endif // INDEXER_REFINE_H
