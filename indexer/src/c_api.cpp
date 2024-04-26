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
#include "cuda_runtime.h"

namespace {
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
            cfg_runtime->length_threshold=1e-9;      // threshold for determining equal vector length (|va| - threshold < |vb| < |va| + threshold)
            cfg_runtime->triml=0.001;                // lower trim value for distance to nearest integer objective value - 0 < triml < trimh
            cfg_runtime->trimh=0.3;                  // higher trim value for distance to nearest integer objective value - triml < trimh < 0.5
            cfg_runtime->delta=0.1;                  // log2 curve position: score = log2(trim(dist(x)) + delta)
            cfg_runtime->dist1=0.2;                  // maximum distance to int for single coordinate
            cfg_runtime->dist3=0.15;                 // maximum distance to int for tripple coordinates
            cfg_runtime->num_halfsphere_points=32*1024; // number of sample points on half sphere for finding vector candidates
            cfg_runtime->num_angle_points=0;         // number of sample points in rotation space for finding cell candidates (0: auto)
        }

        if (cfg_persistent) {
            cfg_persistent->max_output_cells=1;      // maximum number of output unit cells
            cfg_persistent->max_input_cells=1;       // maximum number of input unit cells, (must be before max_spots in memory, see copy_in())
            cfg_persistent->max_spots=200;           // maximum number of input spots, (must be after max_input_cells in memory, see copy_in())
            cfg_persistent->num_candidate_vectors=32; // number of candidate vectors (per input cell vector)
            cfg_persistent->redundant_computations=false; // compute candidates for all three cell vectors instead of just one
        }

        if (cfg_ifssr) {
            cfg_ifssr->threshold_contraction=.8;     // contract error threshold by this value in every iteration
            cfg_ifssr->max_distance=.00075;          // max distance to reciprocal spots for inliers
            cfg_ifssr->min_spots=8;                  // minimum number of spots to fit against
            cfg_ifssr->max_iter=32;                  // max number of iterations
        }
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
            handler_map.emplace(std::make_pair(id, handler{std::move(idx), err, data, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}}));
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
                if (hdl.pin_crt.ptr != cfg_runtime)
                    hdl.pin_crt = fast_feedback::memory_pin::on(cfg_runtime);
                if (hdl.pin_isx.ptr != in->spot.x)
                    hdl.pin_isx = fast_feedback::memory_pin(in->spot.x, in->n_spots * sizeof(float));
                if (hdl.pin_isy.ptr != in->spot.y)
                    hdl.pin_isy = fast_feedback::memory_pin(in->spot.y, in->n_spots * sizeof(float));
                if (hdl.pin_isz.ptr != in->spot.z)
                    hdl.pin_isz = fast_feedback::memory_pin(in->spot.z, in->n_spots * sizeof(float));
                if (hdl.pin_icx.ptr != in->spot.x)
                    hdl.pin_icx = fast_feedback::memory_pin(in->cell.x, 3*in->n_cells * sizeof(float));
                if (hdl.pin_icy.ptr != in->spot.y)
                    hdl.pin_icy = fast_feedback::memory_pin(in->cell.y, 3*in->n_cells * sizeof(float));
                if (hdl.pin_icz.ptr != in->spot.z)
                    hdl.pin_icz = fast_feedback::memory_pin(in->cell.z, 3*in->n_cells * sizeof(float));
                if (hdl.pin_ox.ptr != out->x)
                    hdl.pin_ox = fast_feedback::memory_pin(out->x, 3*out->n_cells * sizeof(float));
                if (hdl.pin_oy.ptr != out->y)
                    hdl.pin_oy = fast_feedback::memory_pin(out->y, 3*out->n_cells * sizeof(float));
                if (hdl.pin_oz.ptr != out->z)
                    hdl.pin_oz = fast_feedback::memory_pin(out->z, 3*out->n_cells * sizeof(float));
                if (hdl.pin_os.ptr != out->score)
                    hdl.pin_os = fast_feedback::memory_pin(out->score, out->n_cells * sizeof(float));
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
                Mx3 cells(3*in->n_cells, 3);
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

    // try to avoid using the following, they don't really belong here
    int num_gpus()
    {
        int ngpus = 0;
        if (cudaGetDeviceCount(&ngpus) != cudaSuccess)
            return -1;
        return ngpus;
    }

    int select_gpu(int gpu)
    {
        if (cudaSetDevice(gpu) != cudaSuccess)
            return -1;
        return 0;
    }
} // extern "C"
