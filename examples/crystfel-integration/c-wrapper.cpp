// Provide a quick and dirty c-wrapper
// for crystfel integration

#include <iostream>
#include <algorithm>
#include <cassert>
#include <ffbidx/refine.h>
#include "ffbidx/c-wrapper.h"

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
            return 1;
        } catch (...) {
            return 1;
        }

        return 0;
    }

    int is_viable_cell(ffbidx_indexer ptr, float cell[9])
    {
        using namespace Eigen;
        using indexer = fast_feedback::refine::indexer<float>;

        int indexable;
        try {
            indexer* idx = (indexer*)ptr.ptr;
            unsigned best =  fast_feedback::refine::best_cell(idx->oScoreV());

            const Matrix<float, 3, 3>& ocell = idx->oCell(best);
            Map<Matrix<float, 3, 3>> mcell{cell};
            mcell = ocell;

            indexable = fast_feedback::refine::is_viable_cell(ocell, idx->Spots());
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

        if (index_step(ptr, cell, x, y, z, nspots) != 0)
            return -1;
        
        return is_viable_cell(ptr, cell);
    }

    int index_refined(ffbidx_indexer ptr, float cell[9], float *x, float *y, float *z, unsigned nspots)
    {
        using indexer = fast_feedback::refine::indexer<float>;
        using ifss = fast_feedback::refine::indexer_ifss<float>;
        using cifss = fast_feedback::refine::config_ifss<float>;

        if (index_step(ptr, cell, x, y, z, nspots) != 0)
            return -1;

        try {
            indexer* idx = (indexer*)ptr.ptr;
            cifss conf_ifss{};

            ifss::refine(idx->Spots(), idx->oCellM(), idx->oScoreV(), conf_ifss);
        } catch (std::exception& ex) {
            std::cerr << "Error: " << ex.what() << '\n';
            return -1;
        } catch (...) {
            return -1;
        }

        return is_viable_cell(ptr, cell);
    }

} // extern "C"

int fast_feedback_crystfel(struct ffbidx_settings *settings, float cell[9], float *x, float *y, float *z, unsigned nspots) {
    using ifss = fast_feedback::refine::indexer_ifss<float>;
    using cifss = fast_feedback::refine::config_ifss<float>;
    using cpers = fast_feedback::config_persistent<float>;
    using crt = fast_feedback::config_runtime<float>;
    using namespace Eigen;

    try {
        cpers cp{};
        crt cr{};
        cifss config_refine{};

        unsigned n = std::min(settings->max_spots, nspots);

        cp.max_input_cells = 1;
        const char* cells_txt = std::getenv("FFBIDX_OUTPUT_CELLS");

        if (cells_txt != nullptr) {
            auto cells = std::strtol(cells_txt, nullptr, 10);
            if ((cells <= 0) || (cells > 64))
                return -1;
            cp.max_output_cells = cells;
        } else
            cp.max_output_cells = 1;
        cp.max_spots = n;

        ifss indexer{cp, cr, config_refine};

        std::copy(&cell[0], &cell[3], &indexer.iCellX());
        std::copy(&cell[3], &cell[6], &indexer.iCellY());
        std::copy(&cell[6], &cell[9], &indexer.iCellZ());

        std::copy(&x[0], &x[n], &indexer.spotX());
        std::copy(&y[0], &y[n], &indexer.spotY());
        std::copy(&z[0], &z[n], &indexer.spotZ());

        indexer.index(1, n);

        auto best_cell_id = fast_feedback::refine::best_cell(indexer.oScoreV());
        auto ocell = indexer.oCell(best_cell_id);

        Map<Matrix<float, 3, 3>> mcell{cell};
        mcell = ocell;
        return fast_feedback::refine::is_viable_cell(ocell, indexer.Spots(), 0.02f, 9u);
    } catch (std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 0;
    } catch (...) {
        std::cerr << "Unknown error: " << '\n';
        return 0;
    }


}
