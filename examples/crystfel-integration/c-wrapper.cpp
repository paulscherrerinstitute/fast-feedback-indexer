// Provide a quick and dirty c-wrapper
// for crystfel integration

#include <iostream>
#include <algorithm>
#include <cassert>
#include <ffbidx/refine.h>
#include "c-wrapper.h"

namespace {

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
        } catch (...) {
            return 1;
        }

        return 0;
    }

    int best_cell(ffbidx_indexer ptr, float cell[9])
    {
        using namespace Eigen;
        using indexer = fast_feedback::refine::indexer<float>;

        try {
            indexer* idx = (indexer*)ptr.ptr;
            const auto& scores = idx->oScoreV();
            auto it = std::min_element(std::cbegin(scores), std::cend(scores));

            Map<Matrix<float, 3, 3>> mcell{cell};
            mcell = idx->oCell(it - std::cbegin(scores));
        } catch (std::exception& ex) {
            std::cerr << "Error: " << ex.what() << '\n';
        } catch (...) {
            return 1;
        }

        return 0;
    }

}

extern "C" {

    int allocate_fast_indexer(ffbidx_indexer* idx, ffbidx_settings* settings)
    {
        using indexer = fast_feedback::refine::indexer<float>;
        using cpers = fast_feedback::config_persistent<float>;
        using crt = fast_feedback::config_runtime<float>;

        idx->ptr = nullptr;

        try {
            cpers cp{};
            crt cr{};
            cp.max_spots = settings->max_spots;
            idx->ptr = new indexer{cp, cr};
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
        using indexer = fast_feedback::refine::indexer<float>;

        int res;
        if ((res = index_step(ptr, cell, x, y, z, nspots)) == 0)
            res = best_cell(ptr, cell);
        return res;
    }

    int index_refined(ffbidx_indexer ptr, float cell[9], float *x, float *y, float *z, unsigned nspots)
    {
        using indexer = fast_feedback::refine::indexer<float>;
        using ifss = fast_feedback::refine::indexer_ifss<float>;
        using cifss = fast_feedback::refine::config_ifss<float>;

        int res;
        if ((res = index_step(ptr, cell, x, y, z, nspots)) != 0)
            return res;

        try {
            indexer* idx = (indexer*)ptr.ptr;
            cifss conf_ifss{};

            ifss::refine(idx->iCoordM(), idx->oCellM(), idx->oScoreV(), idx->conf_persistent(), conf_ifss, idx->n_spots());
        } catch (std::exception& ex) {
            std::cerr << "Error: " << ex.what() << '\n';
        } catch (...) {
            return 1;
        }

        return best_cell(ptr, cell);
    }

} // extern "C"
