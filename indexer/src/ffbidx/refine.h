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
            Eigen::Matrix<float_type, Eigen::Dynamic, 3u> coords;   // coordinates for max_input_cells + max_spots in 3D space
            Eigen::Matrix<float_type, Eigen::Dynamic, 3u> cells;    // max_output_cells coordinate container
            Eigen::Vector<float_type, Eigen::Dynamic> scores;       // output cell scores container
            fast_feedback::input<float_type> input;                 // raw indexer input
            fast_feedback::output<float_type> output;               // raw indexer output
            fast_feedback::config_runtime<float_type> crt;          // raw indexer runtime config
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
                  coords{cp.max_spots + 3u * cp.max_input_cells, 3u},
                  cells{3u * cp.max_output_cells, 3u}, scores{cp.max_output_cells},
                  input{}, output{&cells(0,0), &cells(0,1), &cells(0,2), scores.data(), idx.cpers.max_output_cells},
                  crt{cr},
                  pin_coords{coords}, pin_cells{cells}, pin_scores{scores},
                  pin_crt{fast_feedback::memory_pin::on(crt)}
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
                input.cell.x = &coords(idx.cpers.max_input_cells - n_input_cells,0);
                input.cell.y = &coords(idx.cpers.max_input_cells - n_input_cells,1);
                input.cell.z = &coords(idx.cpers.max_input_cells - n_input_cells,2);
                input.spot.x = &coords(idx.cpers.max_input_cells,0);
                input.spot.y = &coords(idx.cpers.max_input_cells,1);
                input.spot.z = &coords(idx.cpers.max_input_cells,2);
                input.n_cells = n_input_cells;
                input.n_spots = n_spots;
                input.new_cells = true;
                input.new_spots = true;
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

            // Spot access: spot i
            inline float_type& spotX (unsigned i=0u) noexcept
            { return coords(3u * idx.cpers.max_input_cells + i, 0u); }

            inline const float_type& spotX (unsigned i=0u) const noexcept
            { return &coords(3u * idx.cpers.max_input_cells + i, 0u); }

            inline float_type& spotY (unsigned i=0u) noexcept
            { return coords(3u * idx.cpers.max_input_cells + i, 1u); }

            inline const float_type& spotY (unsigned i=0u) const noexcept
            { return coords(3u * idx.cpers.max_input_cells + i, 1u); }

            inline float_type& spotZ (unsigned i=0u) noexcept
            { return coords(3u * idx.cpers.max_input_cells + i, 2u); }

            inline const float_type& spotZ (unsigned i=0u) const noexcept
            { return coords(3u * idx.cpers.max_input_cells + i, 2u); }

            // Coords area designated for spots, fill top down
            inline auto spotM ()
            { return coords.bottomRows(coords.rows() - 3u * idx.cpers.max_input_cells); }

            // Input cell access: cell i, vector j
            inline float_type& iCellX (unsigned i=0u, unsigned j=0u) noexcept
            { return coords(3u * (idx.cpers.max_input_cells - i - 1u) + j, 0u); }

            inline const float_type& iCellX (unsigned i=0u, unsigned j=0u) const noexcept
            { return coords(3u * (idx.cpers.max_input_cells - i - 1u) + j, 0u); }

            inline float_type& iCellY (unsigned i=0u, unsigned j=0u) noexcept
            { return coords(3u * (idx.cpers.max_input_cells - i - 1u) + j, 1u); }

            inline const float_type& iCellY (unsigned i=0u, unsigned j=0u) const noexcept
            { return coords(3u * (idx.cpers.max_input_cells - i - 1u) + j, 1u); }

            inline float_type& iCellZ (unsigned i=0u, unsigned j=0u) noexcept
            { return coords(3u * (idx.cpers.max_input_cells - i - 1u) + j, 2u); }

            inline const float_type& iCellZ (unsigned i=0u, unsigned j=0u) const noexcept
            { return coords(3u * (idx.cpers.max_input_cells - i - 1u) + j, 2u); }

            // Coords area for input cell i
            inline auto iCell (unsigned i=0u) noexcept
            { return coords.block(3u * (idx.cpers.max_input_cells - i - 1u), 0u, 3u, 3u); }

            inline const auto iCell (unsigned i=0u) const noexcept
            { return coords.block(3u * (idx.cpers.max_input_cells - i - 1u), 0u, 3u, 3u); }

            // Coords area designated for input cells, fill cellwise bottom up
            inline auto iCellM () noexcept
            { return coords.topRows(3u * idx.cpers.max_input_cells); }

            // All input coordinates
            inline auto iCoordM () noexcept
            { return coords; }

            // Input size access
            inline unsigned n_input_cells () noexcept
            { return input.n_cells; }

            inline unsigned n_spots () noexcept
            { return input.n_spots; }

            // Output cell access: cell i, vector j
            inline float_type& oCellX (unsigned i=0u, unsigned j=0u) noexcept
            { return cells(3u * i + j, 0u); }

            inline const float_type& oCellX (unsigned i=0u, unsigned j=0u) const noexcept
            { return cells(3u * i + j, 0u); }

            inline float_type& oCellY (unsigned i=0u, unsigned j=0u) noexcept
            { return cells(3u * i + j, 1u); }

            inline const float_type& oCellY (unsigned i=0u, unsigned j=0u) const noexcept
            { return cells(3u * i + j, 1u); }

            inline float_type& oCellZ (unsigned i=0u, unsigned j=0u) noexcept
            { return cells(3u * i + j, 2u); }

            inline const float_type& oCellZ (unsigned i=0u, unsigned j=0u) const noexcept
            { return cells(3u * i + j, 2u); }

            // Output cell i
            inline auto oCell (unsigned i) noexcept
            { return cells.block(3u * i, 0u, 3u, 3u); }

            inline const auto oCell (unsigned i) const noexcept
            { return cells.block(3u * i, 0u, 3u, 3u); }

            // All output cells
            inline auto& oCellM () noexcept
            { return cells; }

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

            const config_runtime<float_type>& conf_runtime () const noexcept
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

            const config_persistent<float_type>& conf_persistent () const noexcept
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
            // - coords     coordinate matrix like the one in the base indexer
            // - cells      output cells matrix like the one in the base indexer
            // - scores     output cell scores matrix with scores coming from the base indexer
            // - cpers      persistent config for the matrices
            // - cifss      ifss config
            // - nspots     actual number of spots in coords
            // - block      which of the N cell blocks
            // - nblocks    use N cell blocks
            // output:
            // - cells      the refined cells
            // - scores     refined cell scores: largest distance of the min_spots closest to their approximated lattice points
            template<typename MatX3, typename VecX>
            inline static void refine (const Eigen::Matrix<float_type, Eigen::Dynamic, 3u>& coords,
                                       Eigen::DenseBase<MatX3>& cells,
                                       Eigen::DenseBase<VecX>& scores,
                                       const fast_feedback::config_persistent<float_type>& cpers,
                                       const config_ifss<float_type>& cifss,
                                       unsigned nspots, unsigned block=0, unsigned nblocks=1)
            {
                using namespace Eigen;
                using Mx3 = Matrix<float_type, Dynamic, 3>;
                using M3 = Matrix<float_type, 3, 3>;
                VectorX<bool> below{nspots};
                MatrixX3<bool> sel{nspots, 3u};
                Mx3 resid{nspots, 3u};
                Mx3 miller{nspots, 3u};
                Mx3 spots = coords.block(3u * cpers.max_input_cells, 0u, nspots, 3u);
                M3 cell;
                const unsigned blocksize = (cpers.max_output_cells + nblocks - 1u) / nblocks;
                const unsigned startcell = block * blocksize;
                const unsigned endcell = std::min(startcell + blocksize, cpers.max_output_cells);
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
                refine(this->coords, this->cells, this->scores, this->idx.cpers, cifss, this->input.n_spots);
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
            // - coords     coordinate matrix like the one in the base indexer
            // - cells      output cells matrix like the one in the base indexer
            // - scores     output cell scores matrix with scores coming from the base indexer
            // - cpers      persistent config for the matrices
            // - cifse      ifse config
            // - nspots     actual number of spots in coords
            // - block      which of the N cell blocks
            // - nblocks    use N cell blocks
            // output:
            // - cells      the refined cells
            // - scores     refined cell scores: largest distance of the min_spots closest to their approximated lattice points
            template<typename MatX3, typename VecX>
            inline static void refine (const Eigen::Matrix<float_type, Eigen::Dynamic, 3u>& coords,
                                       Eigen::DenseBase<MatX3>& cells,
                                       Eigen::DenseBase<VecX>& scores,
                                       const fast_feedback::config_persistent<float_type>& cpers,
                                       const config_ifse<float_type>& cifse,
                                       unsigned nspots, unsigned block=0, unsigned nblocks=1)
            {
                using namespace Eigen;
                using Mx3 = Matrix<float_type, Dynamic, 3>;
                using M3 = Matrix<float_type, 3, 3>;
                VectorX<bool> below{nspots};
                MatrixX3<bool> sel{nspots, 3u};
                Mx3 resid{nspots, 3u};
                Mx3 miller{nspots, 3u};
                Mx3 spots = coords.block(3u * cpers.max_input_cells, 0u, nspots, 3u);
                M3 cell;
                const unsigned blocksize = (cpers.max_output_cells + nblocks - 1u) / nblocks;
                const unsigned startcell = block * blocksize;
                const unsigned endcell = std::min(startcell + blocksize, cpers.max_output_cells);
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
                refine(this->coords, this->cells, this->scores, this->idx.cpers, cifse, this->input.n_spots);
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

    } // namespace refine
} // namespace fast_feedback

#endif // INDEXER_REFINE_H
