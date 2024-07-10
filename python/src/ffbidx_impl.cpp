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

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <Eigen/Core>
#include <iostream>
#include <sstream>
#include <limits>
#include <atomic>
#include <map>
#include <exception>
#include <stdexcept>
#include <string>
#include <memory>
#include "ffbidx/refine.h"

namespace {
    using indexer_t = fast_feedback::indexer<float>;

    std::atomic_uint32_t next_handle{};

    struct map_t {
        indexer_t indexer;
        unsigned n_spots;
        unsigned n_input_cells;
    };

    std::map<uint32_t, map_t> indexers{};

    PyObject* ffbidx_indexer_(PyObject *args, PyObject *kwds)
    {
        using std::numeric_limits;

        constexpr const char* kw[] = {"max_output_cells", "max_input_cells", "max_spots", "num_candidate_vectors", "redundant_computations", nullptr};
        long max_output_cells=32, max_input_cells=1, max_spots=300, num_candidate_vectors=32;
        int redundant_computations=true;
        if (PyArg_ParseTupleAndKeywords(args, kwds, "llll|p", (char**)kw, &max_output_cells, &max_input_cells, &max_spots, &num_candidate_vectors, &redundant_computations) == 0)
            return nullptr;

        if (max_output_cells < 0 || max_output_cells > numeric_limits<unsigned>::max()) {
            PyErr_SetString(PyExc_ValueError, "max_output_cells out of bounds for an unsigned integer");
            return nullptr;
        }
        if (max_input_cells < 0 || max_input_cells > numeric_limits<unsigned>::max()) {
            PyErr_SetString(PyExc_ValueError, "max_input_cells out of bounds for an unsigned integer");
            return nullptr;
        }
        if (max_spots < 0 || max_spots > numeric_limits<unsigned>::max()) {
            PyErr_SetString(PyExc_ValueError, "max_spots out of bounds for an unsigned integer");
            return nullptr;
        }
        if (num_candidate_vectors < 0 || num_candidate_vectors > numeric_limits<unsigned>::max()) {
            PyErr_SetString(PyExc_ValueError, "num_candidate_vectors out of bounds for an unsigned integer");
            return nullptr;
        }
        const fast_feedback::config_persistent<float> cpers{(unsigned)max_output_cells, (unsigned)max_input_cells, (unsigned)max_spots, (unsigned)num_candidate_vectors, (bool)redundant_computations};

        uint32_t handle = (uint32_t)-1;
        try {
            handle = next_handle.fetch_add(1u);
            if (indexers.count(handle) > 0)
                throw std::runtime_error("unable to allocate handle: handle counter wraparound occurred");
            indexers.emplace(handle, map_t{indexer_t{cpers}, 0u, 0u});
        } catch (std::exception& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
            return nullptr;
        }

        return PyLong_FromLong((long)handle);
    }

    PyObject* ffbidx_index_(PyObject *args, PyObject *kwds)
    {
        using std::numeric_limits;

        constexpr const char* kw[] = {"handle",
                                      "spots", "input_cells",
                                      "method",
                                      "length_threshold", "triml", "trimh", "delta", "dist1", "dist3",
                                      "num_halfsphere_points", "num_angle_points", "n_output_cells",
                                      "contraction", "max_dist",
                                      "min_spots", "n_iter",
                                      nullptr};
        long handle;
        PyArrayObject* spots_ndarray = nullptr;
        PyArrayObject* input_cells_ndarray = nullptr;
        const char* method = "ifssr";
        double length_threshold=1e-9, triml=.001, trimh=.3, delta=.1, dist1=.1, dist3=.3;
        long num_halfsphere_points=32*1024, num_angle_points=0, n_output_cells=32;
        double contraction=.8, max_dist=.00075;
        long min_spots=8, n_iter=32;
        if (PyArg_ParseTupleAndKeywords(args, kwds, "lO!O!|sddddddlllddll", (char**)kw,
                                        &handle, &PyArray_Type, &spots_ndarray, &PyArray_Type, &input_cells_ndarray,
                                        &method, &length_threshold, &triml, &trimh, &delta, &dist1, &dist3,
                                        &num_halfsphere_points, &num_angle_points, &n_output_cells,
                                        &contraction, &max_dist, &min_spots, &n_iter) == 0)
            return nullptr;

        if (handle < 0 || handle > numeric_limits<unsigned>::max()) {
            PyErr_SetString(PyExc_ValueError, "handle out of bounds for an unsigned integer");
            return nullptr;
        }

        const std::string smethod{method};
        if ((smethod != "raw") && (smethod != "ifss") && (smethod != "ifse") && (smethod != "ifssr")) {
            PyErr_SetString(PyExc_ValueError, "method must be either raw, ifss, ifse, od ifssr");
            return nullptr;
        }

        if (triml < .0) {
            PyErr_SetString(PyExc_ValueError, "lower trim value < 0");
            return nullptr;
        }

        if (trimh > 0.5) {
            PyErr_SetString(PyExc_ValueError, "higher trim value > 0.5");
            return nullptr;
        }

        if (triml > trimh) {
            PyErr_SetString(PyExc_ValueError, "lower trim value > higher trim value");
            return nullptr;
        }

        if (delta + triml <= .0) {
            PyErr_SetString(PyExc_ValueError, "delta + triml <= 0");
            return nullptr;
        }

        if (dist1 <= .0)
            dist1 = trimh;

        if (dist3 <= .0)
            dist3 = trimh;

        if (num_halfsphere_points < 0 || num_halfsphere_points > numeric_limits<unsigned>::max()) {
            PyErr_SetString(PyExc_ValueError, "num_halfsphere_points out of bounds for an unsigned integer");
            return nullptr;
        }

        if (num_angle_points < 0 || num_angle_points > numeric_limits<unsigned>::max()) {
            PyErr_SetString(PyExc_ValueError, "num_angle_points out of bounds for an unsigned integer");
            return nullptr;
        }
        if (n_output_cells < 0 || n_output_cells > numeric_limits<unsigned>::max()) {
            PyErr_SetString(PyExc_ValueError, "n_output_cells out of bounds for an unsigned integer");
            return nullptr;
        }

        if (contraction <= .0) {
            PyErr_SetString(PyExc_ValueError, "contraction parameter <= 0");
            return nullptr;
        }

        if (max_dist <= .0) {
            PyErr_SetString(PyExc_ValueError, "max distance parameter <= 0");
            return nullptr;
        }

        if (contraction >= 1.) {
            PyErr_SetString(PyExc_ValueError, "contraction parameter >= 1");
            return nullptr;
        }

        if (min_spots < 4 || min_spots > numeric_limits<unsigned>::max()) {
            PyErr_SetString(PyExc_ValueError, "min_spots outside of [4..max_uint]");
            return nullptr;
        }

        if (n_iter < 0 || n_iter > numeric_limits<unsigned>::max()) {
            PyErr_SetString(PyExc_ValueError, "n_iter out of bounds for an unsigned integer");
            return nullptr;
        }

        map_t* entry = nullptr;
        try {
            entry = &indexers.at((unsigned)handle);
        } catch (std::out_of_range&) {
            PyErr_SetString(PyExc_RuntimeError, "invalid handle");
            return nullptr;
        }

        npy_intp n_spots = 0;

        if (PyArray_NDIM(spots_ndarray) != 2) {
            PyErr_SetString(PyExc_RuntimeError, "spots array must be 2 dimensional");
            return nullptr;
        }

        if (PyArray_TYPE(spots_ndarray) != NPY_FLOAT32) {
            PyErr_SetString(PyExc_RuntimeError, "only float32 spot data is supported");
            return nullptr;
        }

        {
            auto* shape = PyArray_DIMS(spots_ndarray);

            if (PyArray_ISCARRAY_RO(spots_ndarray)) {
                if (shape[0] != 3) {
                    PyErr_SetString(PyExc_RuntimeError, "only shape (3, -1) CARRAY spot data is supported");
                    return nullptr;
                }
                n_spots = shape[1];
            } else if (PyArray_ISFARRAY_RO(spots_ndarray)) {
                if (shape[1] != 3) {
                    PyErr_SetString(PyExc_RuntimeError, "only shape (-1, 3) FARRAY spot data is supported");
                    return nullptr;
                }
                n_spots = shape[0];
            } else {
                PyErr_SetString(PyExc_RuntimeError, "only NPY_ARRAY_CARRAY or NPY_ARRAY_FARRAY spot data is supported");
                return nullptr;
            }
        }
        
        if (n_spots <= 0) {
            PyErr_SetString(PyExc_RuntimeError, "no spots");
            return nullptr;
        }

        npy_intp n_input_cells = 0;

        if (PyArray_NDIM(input_cells_ndarray) != 2) {
            PyErr_SetString(PyExc_RuntimeError, "input cells array must be 2 dimensional");
            return nullptr;
        }

        if (PyArray_TYPE(input_cells_ndarray) != NPY_FLOAT32) {
            PyErr_SetString(PyExc_RuntimeError, "only float32 input cell data is supported");
            return nullptr;
        }

        {
            auto* shape = PyArray_DIMS(input_cells_ndarray);

            if (PyArray_ISCARRAY_RO(input_cells_ndarray)) {
                if (shape[0] != 3) {
                    PyErr_SetString(PyExc_RuntimeError, "only shape (3, -1) CARRAY input cell data is supported");
                    return nullptr;
                }
                if (shape[1] % 3 != 0) {
                    PyErr_SetString(PyExc_RuntimeError, "incomplete CARRAY input cell data");
                    return nullptr;
                }
                n_input_cells = shape[1] / 3;
            } else if (PyArray_ISFARRAY_RO(input_cells_ndarray)) {
                if (shape[1] != 3) {
                    PyErr_SetString(PyExc_RuntimeError, "only shape (-1, 3) FARRAY input cell data is supported");
                    return nullptr;
                }
                if (shape[0] % 3 != 0) {
                    PyErr_SetString(PyExc_RuntimeError, "incomplete FARRAY input cell data");
                    return nullptr;
                }
                n_input_cells = shape[0] / 3;
            } else {
                PyErr_SetString(PyExc_RuntimeError, "only NPY_ARRAY_CARRAY or NPY_ARRAY_FARRAY data is supported");
                return nullptr;
            }
        }

        if (n_input_cells <= 0) {
            PyErr_SetString(PyExc_RuntimeError, "no input cells");
            return nullptr;
        }

        unsigned n_out = std::min((unsigned)n_output_cells, entry->indexer.cpers.max_output_cells);

        PyArrayObject* result;
        {
            npy_intp result_dims[] = { 3, 3 * n_out };
            result = (PyArrayObject*)PyArray_SimpleNew(2, result_dims, NPY_FLOAT32);
        }

        if (result == nullptr) {
            PyErr_SetString(PyExc_RuntimeError, "unable to create result array");
            return nullptr;
        }

        PyArrayObject* score;
        {
            npy_intp score_dim = n_out;
            score = (PyArrayObject*)PyArray_SimpleNew(1, &score_dim, NPY_FLOAT32);
        }

        if (score == nullptr) {
            PyErr_SetString(PyExc_RuntimeError, "unable to create score array");
            return nullptr;
        }

        try {
            float* spot_data = (float*)PyArray_DATA(spots_ndarray);
            npy_intp spot_bytes = PyArray_NBYTES(spots_ndarray);

            float* input_cell_data = (float*)PyArray_DATA(input_cells_ndarray);
            npy_intp input_cell_bytes = PyArray_NBYTES(input_cells_ndarray);

            float* out_data = (float*)PyArray_DATA(result);
            npy_intp out_bytes = PyArray_NBYTES(result);

            float* score_data = (float*)PyArray_DATA(score);
            npy_intp score_bytes = PyArray_NBYTES(score);

            fast_feedback::config_runtime<float> crt{(float)length_threshold, (float)triml, (float)trimh, (float)delta, (float)dist1, (float)dist3,
                                                     (unsigned)num_halfsphere_points, (unsigned)num_angle_points};

            fast_feedback::memory_pin pin_crt{fast_feedback::memory_pin::on(crt)};
            fast_feedback::memory_pin pin_score{score_data, (std::size_t)score_bytes};
            fast_feedback::memory_pin pin_out{out_data, (std::size_t)out_bytes};
            fast_feedback::memory_pin pin_spots{spot_data, (std::size_t)spot_bytes};
            fast_feedback::memory_pin pin_input_cells{input_cell_data, (std::size_t)input_cell_bytes};

            const fast_feedback::input<float> input{
                {&input_cell_data[0], &input_cell_data[3*n_input_cells], &input_cell_data[6*n_input_cells]},
                {&spot_data[0], &spot_data[n_spots], &spot_data[2*n_spots]},
                (unsigned)n_input_cells, (unsigned)n_spots,
                true, true
            };
            fast_feedback::output<float> output{&out_data[0], &out_data[3*n_out], &out_data[6*n_out], &score_data[0], n_out};

            entry->indexer.index(input, output, crt);

            entry->n_spots = n_spots;
            entry->n_input_cells = n_input_cells;

            if (smethod != "raw") {
                using namespace Eigen;
                using namespace fast_feedback::refine;
                const Map<MatrixX3f> spots{spot_data, n_spots, 3};
                Map<MatrixX3f> cells{out_data, 3*n_out, 3};
                Map<VectorXf> scores{score_data, n_out};

                if (smethod == "ifss") {
                    config_ifss<float> cifss{(float)contraction, (float)max_dist, (unsigned)min_spots, (unsigned)n_iter};
                    indexer_ifss<float>::refine(spots, cells, scores, cifss);
                } else if (smethod == "ifse") {
                    config_ifse<float> cifse{(float)contraction, (float)max_dist, (unsigned)min_spots, (unsigned)n_iter};
                    indexer_ifse<float>::refine(spots, cells, scores, cifse);
                } else { // ifssr
                    config_ifssr<float> cifssr{(float)contraction, (float)max_dist, (unsigned)min_spots, (unsigned)n_iter};
                    indexer_ifssr<float>::refine(spots, cells, scores, cifssr);
                }
            }
        } catch (std::exception& ex) {
            entry->n_spots = 0u;
            entry->n_input_cells = 0u;
            PyErr_SetString(PyExc_RuntimeError, ex.what());
            return nullptr;
        }

        PyObject* tuple = PyTuple_Pack(2, result, score);
        if (tuple == nullptr)
            PyErr_SetString(PyExc_RuntimeError, "unable to create result tuple");
        return tuple;
    }

    PyObject* ffbidx_crystals_(PyObject *args, PyObject *kwds)
    {
        using std::numeric_limits;

        constexpr const char* kw[] = {"cells", "spots", "scores",
                                      "method",
                                      "threshold",
                                      "min_spots",
                                      nullptr};
        PyArrayObject* cells_ndarray = nullptr;
        PyArrayObject* spots_ndarray = nullptr;
        PyArrayObject* scores_ndarray = nullptr;
        const char* method = "ifssr";
        double threshold=.00075;
        long min_spots=8;

        if (PyArg_ParseTupleAndKeywords(args, kwds, "O!O!O!|sdl", (char**)kw,
                                        &PyArray_Type, &cells_ndarray,
                                        &PyArray_Type, &spots_ndarray,
                                        &PyArray_Type, &scores_ndarray,
                                        &method, &threshold, &min_spots) == 0)
            return nullptr;

        const std::string smethod{method};
        if ((smethod != "raw") && (smethod != "ifss") && (smethod != "ifse") && (smethod != "ifssr")) {
            PyErr_SetString(PyExc_ValueError, "method must be either raw, ifss, ifse, od ifssr");
            return nullptr;
        }

        npy_intp n_cells = 0;

        if (PyArray_NDIM(cells_ndarray) != 2) {
            PyErr_SetString(PyExc_RuntimeError, "cells array must be 2 dimensional");
            return nullptr;
        }

        if (PyArray_TYPE(cells_ndarray) != NPY_FLOAT32) {
            PyErr_SetString(PyExc_RuntimeError, "only float32 cell data is supported");
            return nullptr;
        }

        {
            const auto* shape = PyArray_DIMS(cells_ndarray);

            if (PyArray_ISCARRAY_RO(cells_ndarray)) {
                if (shape[0] != 3) {
                    PyErr_SetString(PyExc_RuntimeError, "only shape (3, -1) CARRAY cell data is supported");
                    return nullptr;
                }
                if (shape[1] % 3 != 0) {
                    PyErr_SetString(PyExc_RuntimeError, "incomplete CARRAY cell data");
                    return nullptr;
                }
                n_cells = shape[1] / 3;
            } else if (PyArray_ISFARRAY_RO(cells_ndarray)) {
                if (shape[1] != 3) {
                    PyErr_SetString(PyExc_RuntimeError, "only shape (-1, 3) FARRAY cell data is supported");
                    return nullptr;
                }
                if (shape[0] % 3 != 0) {
                    PyErr_SetString(PyExc_RuntimeError, "incomplete FARRAY cell data");
                    return nullptr;
                }
                n_cells = shape[0] / 3;
            } else {
                PyErr_SetString(PyExc_RuntimeError, "only NPY_ARRAY_CARRAY or NPY_ARRAY_FARRAY cell data is supported");
                return nullptr;
            }
        }

        if (n_cells <= 0) {
            PyErr_SetString(PyExc_RuntimeError, "no cells");
            return nullptr;
        }

        npy_intp n_spots = 0;

        if (PyArray_NDIM(spots_ndarray) != 2) {
            PyErr_SetString(PyExc_RuntimeError, "spots array must be 2 dimensional");
            return nullptr;
        }

        if (PyArray_TYPE(spots_ndarray) != NPY_FLOAT32) {
            PyErr_SetString(PyExc_RuntimeError, "only float32 spot data is supported");
            return nullptr;
        }

        {
            const auto* shape = PyArray_DIMS(spots_ndarray);

            if (PyArray_ISCARRAY_RO(spots_ndarray)) {
                if (shape[0] != 3) {
                    PyErr_SetString(PyExc_RuntimeError, "only shape (3, -1) CARRAY spot data is supported");
                    return nullptr;
                }
                n_spots = shape[1];
            } else if (PyArray_ISFARRAY_RO(spots_ndarray)) {
                if (shape[1] != 3) {
                    PyErr_SetString(PyExc_RuntimeError, "only shape (-1, 3) FARRAY spot data is supported");
                    return nullptr;
                }
                n_spots = shape[0];
            } else {
                PyErr_SetString(PyExc_RuntimeError, "only NPY_ARRAY_CARRAY or NPY_ARRAY_FARRAY spot data is supported");
                return nullptr;
            }
        }
        
        if (n_spots <= 0) {
            PyErr_SetString(PyExc_RuntimeError, "no spots");
            return nullptr;
        }

        if (PyArray_NDIM(scores_ndarray) != 1) {
            PyErr_SetString(PyExc_RuntimeError, "scores array must be 1 dimensional");
            return nullptr;
        }

        if (PyArray_TYPE(scores_ndarray) != NPY_FLOAT32) {
            PyErr_SetString(PyExc_RuntimeError, "only float32 score data is supported");
            return nullptr;
        }

        {
            const auto* shape = PyArray_DIMS(scores_ndarray);

            if (!PyArray_ISCARRAY_RO(scores_ndarray) && !PyArray_ISFARRAY(scores_ndarray)) {
                PyErr_SetString(PyExc_RuntimeError, "only CARRAY or FARRAY score data is supported");
                    return nullptr;
            }

            if (shape[0] != n_cells) {
                PyErr_SetString(PyExc_RuntimeError, "number of cells and scores must match");
                return nullptr;
            }
        }

        try {
            using namespace Eigen;
            using namespace fast_feedback::refine;

            const Map<const MatrixX3f> cells{(const float*)PyArray_DATA(cells_ndarray), 3*n_cells, 3};
            const Map<const MatrixX3f> spots{(const float*)PyArray_DATA(spots_ndarray), n_spots, 3};
            const Map<const VectorXf> scores{(const float*)PyArray_DATA(scores_ndarray), n_cells};

            const std::vector<unsigned> crystals = select_crystals(cells, spots, scores, (float)threshold, min_spots, smethod=="ifssr");

            if (crystals.empty())
                Py_RETURN_NONE;

            PyArrayObject* crystals_ndarray;
            {
                npy_intp crystals_dim = crystals.size();
                crystals_ndarray = (PyArrayObject*)PyArray_SimpleNew(1, &crystals_dim, NPY_UINT);
            }

            if (crystals_ndarray == nullptr) {
                PyErr_SetString(PyExc_RuntimeError, "unable to create crystals array");
                return nullptr;
            }

            unsigned* crystals_data = (unsigned*)PyArray_DATA(crystals_ndarray);
            std::copy(std::begin(crystals), std::end(crystals), crystals_data);
            return (PyObject*)crystals_ndarray;
        } catch (std::exception& ex) {
            PyErr_SetString(PyExc_RuntimeError, ex.what());
            return nullptr;
        }
    }

    PyObject* ffbidx_release_(PyObject *args, PyObject *kwds)
    {
        using std::numeric_limits;

        constexpr const char* kw[] = {"handle", nullptr};
        long handle;
        if (PyArg_ParseTupleAndKeywords(args, kwds, "l", (char**)kw, &handle) == 0)
            return nullptr;

        if (handle < 0 || handle > numeric_limits<unsigned>::max()) {
            PyErr_SetString(PyExc_ValueError, "handle out of bounds for an unsigned integer");
            return nullptr;
        }

        if (indexers.erase((unsigned)handle) != 1) {
            PyErr_SetString(PyExc_ValueError, "invalid handle");
            return nullptr;
        }

        Py_RETURN_NONE;
    }

    void ffbidx_free(void *)
    {
        indexers.clear();
    }

} // namespace

extern "C" {

    PyObject* ffbidx_indexer([[maybe_unused]] PyObject *self, PyObject *args, PyObject *kwds)
    {
        return ffbidx_indexer_(args, kwds);
    }

    PyObject* ffbidx_index([[maybe_unused]] PyObject *self, PyObject *args, PyObject *kwds)
    {
        return ffbidx_index_(args, kwds);
    }

    PyObject* ffbidx_crystals([[maybe_unused]] PyObject *self, PyObject *args, PyObject *kwds)
    {
        return ffbidx_crystals_(args, kwds);
    }

    PyObject* ffbidx_release([[maybe_unused]] PyObject *self, PyObject *args, PyObject *kwds)
    {
        return ffbidx_release_(args, kwds);
    }

    PyMethodDef ffbidx_methods[] = {
        {"indexer", (PyCFunction)(void*)ffbidx_indexer, METH_VARARGS | METH_KEYWORDS, PyDoc_STR("Get an indexer handle")},
        {"index", (PyCFunction)(void*)ffbidx_index, METH_VARARGS | METH_KEYWORDS, PyDoc_STR("Call indexer")},
        {"crystals", (PyCFunction)(void*)ffbidx_crystals, METH_VARARGS | METH_KEYWORDS, PyDoc_STR("Select crystals")},
        {"release", (PyCFunction)(void*)ffbidx_release, METH_VARARGS | METH_KEYWORDS, PyDoc_STR("Release indexer handle")},
        {NULL, NULL, 0, NULL}
    };

    PyModuleDef ffbidx_module = {
        .m_base = PyModuleDef_HEAD_INIT,
        .m_name = "ffbidx",
        .m_doc = PyDoc_STR("Fast feedback indexer"),
        .m_size = -1,
        .m_methods = ffbidx_methods,
        .m_slots = nullptr,
        .m_traverse = nullptr,
        .m_clear = nullptr,
        .m_free = ffbidx_free
    };

    PyMODINIT_FUNC PyInit_ffbidx_impl(void)
    {
        import_array();
        if (PyErr_Occurred())
            return nullptr;
        PyObject *m = PyModule_Create(&ffbidx_module);
        return m;
    }
    
} // extern "C"
