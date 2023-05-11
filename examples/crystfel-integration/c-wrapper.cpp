// Provide a quick and dirty c-wrapper
// for crystfel integration

#include <iostream>
#include <algorithm>
#include <string>
#include <map>
#include <cstdlib>
#include <cassert>
#include <ffbidx/refine.h>
#include "ffbidx/c-wrapper.h"

namespace {

    struct viable_cell_config {
        float threshold = .02f;
        unsigned n_spots = 9u;
    };

    struct cell_similarity_config {
        float threshold = .02f;
    };

    std::mutex config_mutex;
    bool config_ok = false;

    fast_feedback::config_persistent<float> cpers{};
    fast_feedback::config_runtime<float> crt{};
    fast_feedback::refine::config_ifss<float> cifss{};
    viable_cell_config cvc{};
    cell_similarity_config ccs{};
    unsigned refine = true;
    unsigned redundant_computations = true;

    struct value_t {
        union {
            float f;
            unsigned u;
        };
        char t;
    };

    value_t make_value(const unsigned& val) { value_t v; v.u=val; v.t='u'; return v; }
    value_t make_value(const float& val) { value_t v; v.f=val; v.t='f'; return v; }

    template<typename Out>
    Out& operator<< (Out& out, const value_t& val)
    {
        switch (val.t) {
            case 'f':
                out << val.f; break;
            case 'u':
                out << val.u; break;
            default:
                out << "val/" << val.t; break;
        };
        return out;
    }

    std::map<std::string, value_t> param = {
        {"cpers_max_spots", make_value(200u)},
        {"cpers_max_output_cells", make_value(32u)},
        {"cpers_num_candidate_vectors", make_value(32u)},
        {"cpers_redundant_computations", make_value(redundant_computations)},
        {"crt_num_sample_points", make_value(32u*1024u)},
        {"cifss_min_spots", make_value(6u)},
        {"cvc_threshold", make_value(.02f)},
        {"ccs_threshold", make_value(.02f)},
        {"idx_refine", make_value(refine)},
    };

    template<typename T>
    void parse_val(T& val, const std::string& str)
    {
        if (! (std::istringstream{str} >> val).eof()) {
            std::cerr << "unable to parse: " << str << '\n';
        }
    }

    void set_conf(const ffbidx_settings* settings)
    {
        std::lock_guard config_lock(config_mutex);
        if (config_ok)
            return;

        param["cpers_max_spots"].u = settings->cpers_max_spots;
        param["cpers_max_output_cells"].u = settings->cpers_max_output_cells;
        param["cpers_num_candidate_vectors"].u = settings->cpers_num_candidate_vectors;
        param["cpers_redundant_computations"].u = redundant_computations;
        param["crt_num_sample_points"].u = settings->crt_num_sample_points;
        param["cifss_min_spots"].u = settings->cifss_min_spots;
        param["cvc_threshold"].f = settings->cvc_threshold;
        param["ccs_threshold"].f = ccs.threshold;
        param["idx_refine"].u = refine;

        constexpr const char* pvar_name = "FFBIDX_PARAMS";
        const char* env_c = std::getenv(pvar_name);

        if (env_c != nullptr) {
            std::string env{env_c};
            std::string::size_type start=0, end=0;

            while (end != std::string::npos) {
                if ((end = env.find('=', start)) == std::string::npos) {
                    std::cerr << "unable to parse " << pvar_name << " at " << env.substr(start) << '\n';
                    std::exit(-1);
                };

                std::string key = env.substr(start, end-start);
                auto entry = param.find(key);
                if (entry == param.end()) {
                    std::cerr << pvar_name << " no such key: " << key << '\n';
                    std::exit(-1);
                }

                start = end + 1;
                end = env.find(' ', start);
                std::string val = env.substr(start, end-start);
                switch (entry->second.t) {
                    case 'f':
                        parse_val(entry->second.f, val); break;
                    case 'u':
                        parse_val(entry->second.u, val); break;
                    default:
                        break;
                }

                start = end + 1;
            }
        }

        for (const auto& entry : param) {
            std::cout << entry.first << '=' << entry.second << '\n';
            if (entry.first == "cpers_max_spots") {
                cpers.max_spots = entry.second.u;
            } else if (entry.first == "cpers_max_output_cells") {
                cpers.max_output_cells = entry.second.u;
            } else if (entry.first == "cpers_num_candidate_vectors") {
                cpers.num_candidate_vectors = entry.second.u;
            } else if (entry.first == "cpers_redundant_computations") {
                cpers.redundant_computations = bool(entry.second.u);
            } else if (entry.first == "crt_num_sample_points") {
                crt.num_sample_points = entry.second.u;
            } else if (entry.first == "cifss_min_spots") {
                cifss.min_spots = entry.second.u;
            } else if (entry.first == "cvc_threshold") {
                cvc.threshold = entry.second.f;
            } else if (entry.first == "ccs_threshold") {
                ccs.threshold = entry.second.f;
            } else if (entry.first == "idx_refine") {
                refine = bool(entry.second.u);
            }
        }

        config_ok = true;
    }

    // input:
    // ptr    - convenience indexer object pointer
    // cell   - input cell [x1, x2, x3, y1, y2, y3, z1, z2, z3] in memory
    // x,y,z  - spot coords, size=nspots
    // nspots
    // output:
    // ptr    - convenience indexer object will have the result
    // return:
    // 0 - ok
    // 1 - error
    int index_step(ffbidx_indexer ptr, float cell[9], float *x, float *y, float *z, unsigned nspots)
    {
        using indexer = fast_feedback::refine::indexer<float>;

        try {
            indexer* idx = (indexer*)ptr.ptr;
            std::copy(&cell[0], &cell[3], &idx->iCellX());
            std::copy(&cell[3], &cell[6], &idx->iCellY());
            std::copy(&cell[6], &cell[9], &idx->iCellZ());

            unsigned n = std::min(idx->max_spots(), nspots);
            std::copy(&x[0], &x[n], &idx->spotX());
            std::copy(&y[0], &y[n], &idx->spotY());
            std::copy(&z[0], &z[n], &idx->spotZ());

            idx->index(1u, n);
        } catch (std::exception& ex) {
            std::cerr << "Error: " << ex.what() << '\n';
            return 1;
        } catch (...) {
            return 1;
        }

        return 0;
    }

    // input:
    // ptr  - convenience indexer object pointer
    // cell - given cell [x1, x2, x3, y1, y2, y3, z1, z2, z3] in memory
    // output:
    // cell - best cell [x1, x2, x3, y1, y2, y3, z1, z2, z3] in memory
    // return:
    // 0 - cell not viable (frame not indexable)
    // 1 - cell viable (frame indexable)
    int viable_cell(ffbidx_indexer ptr, float cell[9])
    {
        using namespace Eigen;
        using indexer = fast_feedback::refine::indexer<float>;

        int indexable;
        try {
            indexer* idx = (indexer*)ptr.ptr;
            Map<Matrix<float, 3, 3>> mcell{cell};

            for (unsigned i=0u; i<cpers.max_output_cells; i++)
                idx->oScore(i) += fast_feedback::refine::cell_similarity(idx->oCell(i), mcell, ccs.threshold);

            unsigned best =  fast_feedback::refine::best_cell(idx->oScoreV());

            const Matrix<float, 3, 3>& ocell = idx->oCell(best);
            mcell = ocell;

            indexable = fast_feedback::refine::is_viable_cell(ocell, idx->Spots(), cvc.threshold, cvc.n_spots);
        } catch (std::exception& ex) {
            std::cerr << "Error: " << ex.what() << '\n';
            return -1;
        } catch (...) {
            return -1;
        }

        return indexable;
    }

}

extern "C" {

    int allocate_fast_indexer(ffbidx_indexer* idx, ffbidx_settings* settings)
    {
        using indexer = fast_feedback::refine::indexer<float>;

        idx->ptr = nullptr;

        try {
            set_conf(settings);
            idx->ptr = new indexer{cpers, crt};
        } catch (std::exception& ex) {
            std::cerr << "Error: " << ex.what() << '\n';
        } catch (...) {
            // ignore
        }

        return idx->ptr == nullptr;
    }

    void free_fast_indexer(ffbidx_indexer ptr)
    {
        using indexer = fast_feedback::refine::indexer<float>;

        indexer* idx = (indexer*)ptr.ptr;
        delete idx;
        ptr.ptr = nullptr;
    }

    // cell: 0-3: x-coords, 3-6: y-coords, 6-9: z-coords
    int index_raw(ffbidx_indexer ptr, float cell[9], float *x, float *y, float *z, unsigned nspots)
    {
        if (index_step(ptr, cell, x, y, z, nspots) != 0)
            return -1;
        
        return viable_cell(ptr, cell);
    }

    // cell: 0-3: x-coords, 3-6: y-coords, 6-9: z-coords
    int index_refined(ffbidx_indexer ptr, float cell[9], float *x, float *y, float *z, unsigned nspots)
    {
        using indexer = fast_feedback::refine::indexer<float>;
        using ifss = fast_feedback::refine::indexer_ifss<float>;

        if (index_step(ptr, cell, x, y, z, nspots) != 0)
            return -1;

        try {
            indexer* idx = (indexer*)ptr.ptr;

            ifss::refine(idx->Spots(), idx->oCellM(), idx->oScoreV(), cifss);
        } catch (std::exception& ex) {
            std::cerr << "Error: " << ex.what() << '\n';
            return -1;
        } catch (...) {
            return -1;
        }

        return viable_cell(ptr, cell);
    }

    // cell: 0-3: x-coords, 3-6: y-coords, 6-9: z-coords
    int fast_feedback_crystfel(struct ffbidx_settings *settings, float cell[9], float *x, float *y, float *z, unsigned nspots) {
        using namespace Eigen;

        ffbidx_indexer idx;
        int res;

        if (allocate_fast_indexer(&idx, settings) != 0)
            return -1;
        res = refine ?
                index_refined(idx, cell, x, y, z, nspots) :
                index_raw(idx, cell, x, y, z, nspots);
        free_fast_indexer(idx);

        Map<Matrix<float, 3, 3>> mcell{cell};
        fast_feedback::refine::make_right_handed(mcell);

        return res;
    }

} // extern "C"
