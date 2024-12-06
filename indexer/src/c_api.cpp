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

#include <unordered_map>
#include <stdexcept>
#include <atomic>
#include <limits>
#include <memory>
#include <cstdio>
#include <cstring>
#include <utility>
#include <algorithm>
#include <Eigen/Dense>
#include "ffbidx/refine.h"
#include "ffbidx/c_api.h"

namespace {
    // Make sure the config and input/output structs are compatible between C and C++
    static_assert(sizeof(fast_feedback::config_persistent<float>) == sizeof(::config_persistent));
    static_assert(sizeof(fast_feedback::config_runtime<float>) == sizeof(::config_runtime));
    static_assert(sizeof(fast_feedback::refine::config_ifssr<float>) == sizeof(::config_ifssr));
    static_assert(sizeof(fast_feedback::input<float>) == sizeof(::input));
    static_assert(sizeof(fast_feedback::output<float>) == sizeof(::output));

    constexpr std::string_view no_error{"OK"};

    struct handler final {
        std::unique_ptr<fast_feedback::indexer<float_type>> idx;
        error* err;
        void* data;
        fast_feedback::memory_pin pin_isx;
        fast_feedback::memory_pin pin_isy;
        fast_feedback::memory_pin pin_isz;
        fast_feedback::memory_pin pin_icx;
        fast_feedback::memory_pin pin_icy;
        fast_feedback::memory_pin pin_icz;
        fast_feedback::memory_pin pin_ox;
        fast_feedback::memory_pin pin_oy;
        fast_feedback::memory_pin pin_oz;
        fast_feedback::memory_pin pin_os;
        fast_feedback::memory_pin pin_crt;
        bool pin_dynamic;
    };

    std::atomic_int next_id{0};
    std::unordered_map<int, handler> handler_map{};

    void set_error(error* err, const char* message)
    {
        if (err)
            std::snprintf(err->message, err->msg_len, "%s", message);
    }

    int error_from_exception(handler& hdl, const std::exception& ex)
    {
        set_error(hdl.err, ex.what());
        return -1;
    }

    int error_from_msg(handler& hdl, const std::string_view& msg)
    {
        set_error(hdl.err, msg.data());
        return -1;
    }

    void set_defaults_impl(config_persistent* cfg_persistent,
                           config_runtime* cfg_runtime,
                           config_ifssr* cfg_ifssr)
    {
        if (cfg_runtime) {
            const fast_feedback::config_runtime<float> crt{};
            std::memcpy(cfg_runtime, reinterpret_cast<const config_runtime*>(&crt), sizeof(config_runtime));
        }

        if (cfg_persistent) {
            const fast_feedback::config_persistent<float> cpers{};
            std::memcpy(cfg_persistent, &cpers, sizeof(config_persistent));
        }

        if (cfg_ifssr) {
            const fast_feedback::refine::config_ifssr<float> cifssr;
            std::memcpy(cfg_ifssr, &cifssr, sizeof(config_ifssr));
        }
    }

    int create_indexer_impl(const config_persistent* cfg_persistent,
                            bool pin_dynamic,
                            error* err,
                            void* data)
    {
        static_assert(sizeof(fast_feedback::config_persistent<float_type>) == sizeof(*cfg_persistent));
        try {
            int id = next_id.fetch_add(1);
            if (id < 0) {
                next_id.store(std::numeric_limits<int>::min());
                return -1;
            }
            const auto* cpers = reinterpret_cast<const fast_feedback::config_persistent<float_type>*>(cfg_persistent);
            std::unique_ptr<fast_feedback::indexer<float_type>> idx{new fast_feedback::indexer<float_type>{*cpers}};
            handler_map.emplace(std::make_pair(id, handler{std::move(idx), err, data, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, pin_dynamic}));
            error_from_msg(handler_map[id], no_error);
            return id;
        } catch (std::exception& ex) {
            set_error(err, ex.what());
        } catch (...) {
            ; // ignore
        }
        return -1;
    }

    int drop_indexer_impl(int handle)
    {   
        struct error* err = NULL;
        try {
            handler& hdl = handler_map.at(handle);
            err = hdl.err;
            return handler_map.erase(handle) != 0 ? 0 : -1;
        } catch (std::exception& ex) {
            set_error(err, ex.what());
        } catch (...) {
            ; // ignore
        }
        return -1;
    }

    error* error_message_impl(int handle)
    {
        try {
            handler& hdl = handler_map.at(handle);
            return hdl.err;
        } catch(...) {
            return NULL;
        }
    }

    void* user_data_impl(int handle)
    {
        try {
            handler& hdl = handler_map.at(handle);
            return hdl.data;
        } catch(...) {
            return NULL;
        }
    }

    int index_start_impl(int handle,
                         const input* in,
                         output* out,
                         const config_runtime* cfg_runtime,
                         callback_func callback,
                         void* data)
    {
        static_assert(sizeof(fast_feedback::input<float_type>) == sizeof(*in));
        static_assert(sizeof(fast_feedback::output<float_type>) == sizeof(*out));
        const fast_feedback::config_runtime<float_type> default_crt{};
        try {
            handler& hdl = handler_map.at(handle);
            try {
                const fast_feedback::config_persistent<float>& cpers = hdl.idx->cpers;
                if (hdl.pin_crt.ptr != cfg_runtime)
                    hdl.pin_crt = fast_feedback::memory_pin::on(cfg_runtime);
                if (hdl.pin_isx.ptr != in->spot.x)
                    hdl.pin_isx = fast_feedback::memory_pin(in->spot.x, (hdl.pin_dynamic ? in->n_spots : cpers.max_spots) * sizeof(float));
                if (hdl.pin_isy.ptr != in->spot.y)
                    hdl.pin_isy = fast_feedback::memory_pin(in->spot.y, (hdl.pin_dynamic ? in->n_spots : cpers.max_spots) * sizeof(float));
                if (hdl.pin_isz.ptr != in->spot.z)
                    hdl.pin_isz = fast_feedback::memory_pin(in->spot.z, (hdl.pin_dynamic ? in->n_spots : cpers.max_spots) * sizeof(float));
                if (hdl.pin_icx.ptr != in->cell.x)
                    hdl.pin_icx = fast_feedback::memory_pin(in->cell.x, 3*(hdl.pin_dynamic ? in->n_cells : cpers.max_input_cells) * sizeof(float));
                if (hdl.pin_icy.ptr != in->cell.y)
                    hdl.pin_icy = fast_feedback::memory_pin(in->cell.y, 3*(hdl.pin_dynamic ? in->n_cells : cpers.max_input_cells) * sizeof(float));
                if (hdl.pin_icz.ptr != in->cell.z)
                    hdl.pin_icz = fast_feedback::memory_pin(in->cell.z, 3*(hdl.pin_dynamic ? in->n_cells : cpers.max_input_cells) * sizeof(float));
                if (hdl.pin_ox.ptr != out->x)
                    hdl.pin_ox = fast_feedback::memory_pin(out->x, 3*(hdl.pin_dynamic ? out->n_cells : cpers.max_output_cells) * sizeof(float));
                if (hdl.pin_oy.ptr != out->y)
                    hdl.pin_oy = fast_feedback::memory_pin(out->y, 3*(hdl.pin_dynamic ? out->n_cells : cpers.max_output_cells) * sizeof(float));
                if (hdl.pin_oz.ptr != out->z)
                    hdl.pin_oz = fast_feedback::memory_pin(out->z, 3*(hdl.pin_dynamic ? out->n_cells : cpers.max_output_cells) * sizeof(float));
                if (hdl.pin_os.ptr != out->score)
                    hdl.pin_os = fast_feedback::memory_pin(out->score, (hdl.pin_dynamic ? out->n_cells : cpers.max_output_cells) * sizeof(float));
                const auto* crt = cfg_runtime == nullptr ?
                    &default_crt :
                    reinterpret_cast<const fast_feedback::config_runtime<float_type>*>(cfg_runtime);
                hdl.idx->index_start(
                    *reinterpret_cast<const fast_feedback::input<float_type>*>(in),
                    *reinterpret_cast<fast_feedback::output<float_type>*>(out),
                    *crt,
                    callback, data
                );
            } catch (std::exception& ex) {
                return error_from_exception(hdl, ex);
            }
        } catch (...) {
            return -1;
        }
        return 0;
    }

    int index_end_impl(int handle,
                       output* out)
    {
        try {
            handler& hdl = handler_map.at(handle);
            try {
                hdl.idx->index_end(
                    *reinterpret_cast<fast_feedback::output<float_type>*>(out)
                );
            } catch (std::exception& ex) {
                return error_from_exception(hdl, ex);
            }
        } catch (...) {
            return -1;
        }
        return 0;
    }

    int refine_impl(int handle,
                    const input* in,
                    output* out,
                    const config_ifssr* cfg_ifssr,
                    unsigned block, unsigned nblocks)
    {
        namespace refine = fast_feedback::refine;
        using Mx3 = Eigen::MatrixX3<float_type>;
        using Vx = Eigen::VectorXf;
        using Eigen::Map;

        if (cfg_ifssr == NULL)
            return 0;

        try {
            handler& hdl = handler_map.at(handle);
            try {
                Mx3 spots{in->n_spots, 3};
                const Map<Vx> sx{in->spot.x, in->n_spots};
                spots.col(0) = sx;
                const Map<Vx> sy{in->spot.y, in->n_spots};
                spots.col(1) = sy;
                const Map<Vx> sz{in->spot.z, in->n_spots};
                spots.col(2) = sz;
                Mx3 cells(3*out->n_cells, 3);
                Map<Vx> cx{out->x, 3*out->n_cells};
                cells.col(0) = cx;
                Map<Vx> cy{out->y, 3*out->n_cells};
                cells.col(1) = cy;
                Map<Vx> cz{out->z, 3*out->n_cells};
                cells.col(2) = cz;
                Map<Vx> sc{out->score, out->n_cells};
                Vx scores{sc};
                static_assert(sizeof(refine::config_ifssr<float_type>) == sizeof(config_ifssr));
                refine::indexer_ifssr<float_type>::refine(
                    spots, cells, scores,
                    *reinterpret_cast<const refine::config_ifssr<float_type>*>(cfg_ifssr),
                    block, nblocks
                );
                cx = cells.col(0);
                cy = cells.col(1);
                cz = cells.col(2);
                sc = scores;
            } catch (std::exception& ex) {
                return error_from_exception(hdl, ex);
            }
        } catch (...) {
            return -1;
        }
        return 0;
    }

    int indexer_op_impl(int handle,
                        const input* in,
                        output* out,
                        const config_runtime* cr,
                        const config_ifssr* cfg_ifssr)
    {
        if (index_start_impl(handle, in, out, cr, nullptr, nullptr))
            return -1;
        if (index_end_impl(handle, out))
            return -1;
        if (refine_impl(handle, in, out, cfg_ifssr, 0, 1))
            return -1;
        int res;
        if ((res = best_cell(handle, out)) < 0)
            return -1;
        return res;
    }

    int best_cell_impl(int handle,
                       const output* out)
    {
        namespace refine = fast_feedback::refine;
        using Vx = Eigen::VectorXf;
        using Eigen::Map;
        try {
            handler& hdl = handler_map.at(handle);
            try {
                Vx scores{Map<Vx>{out->score, out->n_cells}};
                return (int)refine::best_cell(scores);
            } catch (std::exception& ex) {
                return error_from_exception(hdl, ex);
            }
        } catch (...) {
            return -1;
        }
    }

    int crystals_impl(int handle,
                      const input* in,
                      const output* out,
                      float_type threshold,
                      unsigned min_spots,
                      unsigned* indices,
                      unsigned indices_size)
    {
        namespace refine = fast_feedback::refine;
        using Mx3 = Eigen::MatrixX3<float_type>;
        using Vx = Eigen::VectorXf;
        using Eigen::Map;
        try {
            handler& hdl = handler_map.at(handle);
            try {
                Mx3 spots(in->n_spots, 3);
                spots.col(0) = Map<Vx>(in->spot.x, in->n_spots);
                spots.col(1) = Map<Vx>(in->spot.y, in->n_spots);
                spots.col(2) = Map<Vx>(in->spot.z, in->n_spots);
                Mx3 cells(3*out->n_cells, 3);
                cells.col(0) = Map<Vx>(out->x, 3*out->n_cells);
                cells.col(1) = Map<Vx>(out->y, 3*out->n_cells);
                cells.col(2) = Map<Vx>(out->z, 3*out->n_cells);
                Vx scores{Map<Vx>{out->score, out->n_cells}};
                auto idx = refine::select_crystals(
                    cells, spots,
                    scores,
                    threshold, min_spots
                );
                unsigned len = std::min((unsigned)idx.size(), indices_size);
                std::copy(std::cbegin(idx), std::cbegin(idx)+len, indices);
                return len;
            } catch (std::exception& ex) {
                return error_from_exception(hdl, ex);
            }
        } catch (...) {
            return -1;
        }
    }
} // namespace

extern "C" {
    void set_defaults(config_persistent* cfg_persistent,
                      config_runtime* cfg_runtime,
                      config_ifssr* cfg_ifssr)
    {
        set_defaults_impl(cfg_persistent, cfg_runtime, cfg_ifssr);
    }

    int create_indexer(const config_persistent* cfg_persistent,
                       bool pin_dynamic,
                       error* err,
                       void* data)
    {
        return create_indexer_impl(cfg_persistent, pin_dynamic, err, data);
    }

    int drop_indexer(int handle)
    {
        return drop_indexer_impl(handle);
    }

    error* error_message(int handle)
    {
        return error_message_impl(handle);
    }

    void* user_data(int handle)
    {
        return user_data_impl(handle);
    }

    int index_start(int handle,
                    const input* in,
                    output* out,
                    const config_runtime* cfg_runtime,
                    callback_func callback,
                    void* data)
    {
        return index_start_impl(handle, in, out, cfg_runtime, callback, data);
    }

    int index_end(int handle,
                  output* out)
    {
        return index_end_impl(handle, out);
    }

    int refine(int handle,
               const input* in,
               output* out,
               const config_ifssr* cfg_ifssr,
               unsigned block, unsigned nblocks)
    {
        return refine_impl(handle, in, out, cfg_ifssr, block, nblocks);
    }

    int indexer_op(int handle,
                   const input* in,
                   output* out,
                   const config_runtime* cfg_runtime,
                   const config_ifssr* cfg_ifssr)
    {
        return indexer_op_impl(handle, in, out, cfg_runtime, cfg_ifssr);
    }

    int best_cell(int handle,
                  const output* out)
    {
        return best_cell_impl(handle, out);
    }

    int crystals(int handle,
                 const input* in,
                 const output* out,
                 float_type threshold,
                 unsigned min_spots,
                 unsigned* indices,
                 unsigned indices_size)
    {
        return crystals_impl(handle, in, out, threshold, min_spots, indices, indices_size);
    }

    int check_config(const struct config_persistent* cfg_persistent,
                     const struct config_runtime* cfg_runtime,
                     const struct config_ifssr* cfg_ifssr,
                     struct error* err)
    {
        try {
            fast_feedback::check_config(
                reinterpret_cast<const fast_feedback::config_persistent<float>*>(cfg_persistent),
                reinterpret_cast<const fast_feedback::config_runtime<float>*>(cfg_runtime)
            );
            if (cfg_ifssr)
                fast_feedback::refine::indexer_ifssr<float>::check_config(
                    *reinterpret_cast<const fast_feedback::refine::config_ifssr<float>*>(cfg_ifssr)
                );
            return 0;
        } catch (std::exception& ex) {
            set_error(err, ex.what());
        } catch (...) {
            ; // ignore
        }
        return -1;
    }

    // try to avoid using the following, they don't really belong here
    extern int _num_gpus();         // from indexer_gpu.cu
    extern int _select_gpu(int);    // from indexer_gpu.cu

    int num_gpus()
    {
        return _num_gpus();
    }

    int select_gpu(int gpu)
    {
        return _select_gpu(gpu);
    }
} // extern "C"
