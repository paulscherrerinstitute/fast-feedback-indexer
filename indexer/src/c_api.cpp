#include <unordered_map>
#include <stdexcept>
#include <atomic>
#include <limits>
#include <memory>
#include <cstdio>
#include <utility>
#include <algorithm>
#include <Eigen/Dense>
#include "ffbidx/refine.h"
#include "ffbidx/c_api.h"

namespace {
    constexpr std::string_view no_error{"OK"};

    struct handler final {
        std::unique_ptr<fast_feedback::indexer<float_type>> idx;
        error* err;
        void* data;
    };

    std::atomic_int next_id{0};
    std::unordered_map<int, handler> handler_map{};

    int error_from_exception(handler& hdl, const std::exception& ex)
    {
        if (hdl.err)
            std::snprintf(hdl.err->message, hdl.err->msg_len, "%s", ex.what());
        return -1;
    }

    int error_from_msg(handler& hdl, const std::string_view& msg)
    {
        if (hdl.err)
            std::snprintf(hdl.err->message, hdl.err->msg_len, "%s", msg.data());
        return -1;
    }

    int create_indexer_impl(const config_persistent* cfg_persistent,
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
            handler_map.emplace(std::make_pair(id, handler{std::move(idx), err, data}));
            error_from_msg(handler_map[id], no_error);
            return id;
        } catch (...) {
            return -1;
        }
    }

    int drop_indexer_impl(int handle)
    {
        return handler_map.erase(handle) != 0 ? 0 : -1;
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
            const auto* crt = cfg_runtime == nullptr ?
                &default_crt :
                reinterpret_cast<const fast_feedback::config_runtime<float_type>*>(cfg_runtime);
            try {
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

        if (! cfg_ifssr)
            return 0;

        try {
            handler& hdl = handler_map.at(handle);
            try {
                Mx3 spots(in->n_spots, 3);
                spots.col(0) = Map<Vx>(in->spot.x, in->n_spots);
                spots.col(1) = Map<Vx>(in->spot.y, in->n_spots);
                spots.col(2) = Map<Vx>(in->spot.z, in->n_spots);
                Mx3 cells(3*in->n_cells, 3);
                cells.col(0) = Map<Vx>(out->x, 3*out->n_cells);
                cells.col(1) = Map<Vx>(out->y, 3*out->n_cells);
                cells.col(2) = Map<Vx>(out->z, 3*out->n_cells);
                Vx scores{Map<Vx>{out->score, out->n_cells}};
                static_assert(sizeof(refine::config_ifssr<float_type>) == sizeof(config_ifssr));
                refine::indexer_ifssr<float_type>::refine(
                    spots, cells, scores,
                    *reinterpret_cast<const refine::config_ifssr<float_type>*>(cfg_ifssr),
                    block, nblocks
                );
            } catch (std::exception& ex) {
                return error_from_exception(hdl, ex);
            }
        } catch (...) {
            return -1;
        }
        return 0;
    }

    int index_impl(int handle,
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
                Mx3 cells(3*in->n_cells, 3);
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
                return 0;
            } catch (std::exception& ex) {
                return error_from_exception(hdl, ex);
            }
        } catch (...) {
            return -1;
        }
    }
} // namespace

extern "C" {
    int create_indexer(const config_persistent* cfg_persistent,
                       error* err,
                       void* data)
    {
        return create_indexer_impl(cfg_persistent, err, data);
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

    int index(int handle,
              const input* in,
              output* out,
              const config_runtime* cfg_runtime,
              const config_ifssr* cfg_ifssr)
    {
        return index_impl(handle, in, out, cfg_runtime, cfg_ifssr);
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
} // extern "C"
