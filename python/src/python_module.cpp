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
#include "ffbidx/refine.h"

namespace {
    using indexer_t = fast_feedback::indexer<float>;

    std::atomic_uint32_t next_handle{};

    std::map<uint32_t, indexer_t> indexers{};

    PyObject* ffbidx_indexer_(PyObject *args, PyObject *kwds)
    {
        using std::numeric_limits;

        constexpr const char* kw[] = {"max_output_cells", "max_input_cells", "max_spots", "num_candidate_vectors", nullptr};
        long max_output_cells, max_input_cells, max_spots, num_candidate_vectors;
        if (PyArg_ParseTupleAndKeywords(args, kwds, "llll", (char**)kw, &max_output_cells, &max_input_cells, &max_spots, &num_candidate_vectors) == 0)
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
        const fast_feedback::config_persistent<float> cpers{(unsigned)max_output_cells, (unsigned)max_input_cells, (unsigned)max_spots, (unsigned)num_candidate_vectors};

        uint32_t handle = (uint32_t)-1;
        try {
            handle = next_handle.fetch_add(1u);
            if (indexers.count(handle) > 0)
                throw std::runtime_error("unable to allocate handle: handle counter wraparound occurred");
            indexers.emplace(handle, indexer_t{cpers});
        } catch (std::exception& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
            return nullptr;
        }

        return PyLong_FromLong((long)handle);
    }

    PyObject* ffbidx_index_(PyObject *args, PyObject *kwds)
    {
        using std::numeric_limits;

        constexpr const char* kw[] = {"handle", "data",
                                      "method", "length_threshold", "triml", "trimh", "delta", "num_sample_points", "n_output_cells", "n_input_cells",
                                      "contraction", "min_spots", "n_iter",
                                      nullptr};
        long handle;
        PyArrayObject* ndarray;
        const char* method = "ifss";
        double length_threshold=1e-9, triml=.05, trimh=.15, delta=.1;
        long num_sample_points=32*1024, n_output_cells=1, n_input_cells=1;
        double contraction=.8;
        long min_spots=6, n_iter=15;
        if (PyArg_ParseTupleAndKeywords(args, kwds, "lO!|sddddllldll", (char**)kw, &handle, &PyArray_Type, &ndarray,
                                        &method, &length_threshold, &triml, &trimh, &delta, &num_sample_points, &n_output_cells, &n_input_cells,
                                        &contraction, &min_spots, &n_iter) == 0)
            return nullptr;

        if (handle < 0 || handle > numeric_limits<unsigned>::max()) {
            PyErr_SetString(PyExc_ValueError, "handle out of bounds for an unsigned integer");
            return nullptr;
        }

        const std::string smethod{method};
        if ((smethod != "raw") && (smethod != "ifss") && (smethod != "ifse")) {
            PyErr_SetString(PyExc_ValueError, "method must be either raw, ifss, or ifse");
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

        if (num_sample_points < 0 || num_sample_points > numeric_limits<unsigned>::max()) {
            PyErr_SetString(PyExc_ValueError, "num_sample_points out of bounds for an unsigned integer");
            return nullptr;
        }

        if (n_output_cells < 0 || n_output_cells > numeric_limits<unsigned>::max()) {
            PyErr_SetString(PyExc_ValueError, "n_output_cells out of bounds for an unsigned integer");
            return nullptr;
        }

        if (n_input_cells < 0 || n_input_cells > numeric_limits<unsigned>::max()) {
            PyErr_SetString(PyExc_ValueError, "n_input_cells out of bounds for an unsigned integer");
            return nullptr;
        }

        if (contraction <= .0) {
            PyErr_SetString(PyExc_ValueError, "contraction parameter <= 0");
            return nullptr;
        }

        if (smethod == "ifss" && contraction >= 1.) {
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

        if (PyArray_NDIM(ndarray) != 2) {
            PyErr_SetString(PyExc_RuntimeError, "data array must be 2 dimensional");
            return nullptr;
        }

        if (PyArray_TYPE(ndarray) != NPY_FLOAT32) {
            PyErr_SetString(PyExc_RuntimeError, "only float32 data is supported");
            return nullptr;
        }

        npy_intp n_vecs = 0;
        {
            auto* shape = PyArray_DIMS(ndarray);

            if (PyArray_ISCARRAY(ndarray)) {
                if (shape[0] != 3) {
                    PyErr_SetString(PyExc_RuntimeError, "only shape (3, -1) CARRAY data is supported");
                    return nullptr;
                }
                n_vecs = shape[1];
            } else if (PyArray_ISFARRAY(ndarray)) {
                if (shape[1] != 3) {
                    PyErr_SetString(PyExc_RuntimeError, "only shape (-1, 3) FARRAY data is supported");
                    return nullptr;
                }
                n_vecs = shape[0];
            } else {
                PyErr_SetString(PyExc_RuntimeError, "only NPY_ARRAY_CARRAY or NPY_ARRAY_FARRAY data is supported");
                return nullptr;
            }
        }

        if (3*n_input_cells >= n_vecs) {
            PyErr_SetString(PyExc_RuntimeError, "not enough data for n_cells plus spots");
            return nullptr;
        }

        indexer_t* indexer = nullptr;
        try {
            indexer = &indexers.at((unsigned)handle);
        } catch (std::out_of_range&) {
            PyErr_SetString(PyExc_RuntimeError, "invalid handle");
            return nullptr;
        }

        unsigned n_out = std::min((unsigned)n_output_cells, indexer->cpers.max_output_cells);

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
            float* in_data = (float*)PyArray_DATA(ndarray);
            npy_intp in_bytes = PyArray_NBYTES(ndarray);

            float* out_data = (float*)PyArray_DATA(result);
            npy_intp out_bytes = PyArray_NBYTES(result);

            float* score_data = (float*)PyArray_DATA(score);
            npy_intp score_bytes = PyArray_NBYTES(score);

            fast_feedback::config_runtime<float> crt{(float)length_threshold, (float)triml, (float)trimh, (float)delta, (unsigned)num_sample_points};

            fast_feedback::memory_pin pin_crt{fast_feedback::memory_pin::on(crt)};
            fast_feedback::memory_pin pin_score{score_data, (std::size_t)score_bytes};
            fast_feedback::memory_pin pin_out{out_data, (std::size_t)out_bytes};
            fast_feedback::memory_pin pin_in{in_data, (std::size_t)in_bytes};

            unsigned n_spots = n_vecs - 3*n_input_cells;
            const fast_feedback::input<float> input{
                {&in_data[0], &in_data[n_vecs], &in_data[2*n_vecs]},
                {&in_data[3*n_input_cells], &in_data[n_vecs + 3*n_input_cells], &in_data[2*n_vecs + 3*n_input_cells]},
                (unsigned)n_input_cells, n_spots,
                true, true
            };
            fast_feedback::output<float> output{&out_data[0], &out_data[3*n_out], &out_data[6*n_out], &score_data[0], n_out};

            indexer->index(input, output, crt);

            if (smethod != "raw") {
                    using namespace Eigen;
                    using namespace fast_feedback::refine;
                    Map<MatrixX3f> coords{in_data, n_vecs, 3};
                    Map<MatrixX3f> cells{out_data, 3*n_out, 3};
                    Map<VectorXf> scores{score_data, n_out};

                if (smethod == "ifss") {
                    config_ifss<float> cifss{(float)contraction, (unsigned)min_spots, (unsigned)n_iter};
                    indexer_ifss<float>::refine(coords, cells, scores, indexer->cpers, cifss, n_spots);
                } else { // ifse
                    config_ifse<float> cifse{(float)contraction, (unsigned)min_spots, (unsigned)n_iter};
                    indexer_ifse<float>::refine(coords, cells, scores, indexer->cpers, cifse, n_spots);
                }
            }
        } catch (std::exception& ex) {
            PyErr_SetString(PyExc_RuntimeError, ex.what());
            return nullptr;
        }

        PyObject* tuple = PyTuple_Pack(2, result, score);
        if (tuple == nullptr)
            PyErr_SetString(PyExc_RuntimeError, "unable to create result tuple");
        return tuple;
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

    PyObject* ffbidx_release([[maybe_unused]] PyObject *self, PyObject *args, PyObject *kwds)
    {
        return ffbidx_release_(args, kwds);
    }

    PyMethodDef ffbidx_methods[] = {
        {"indexer", (PyCFunction)(void*)ffbidx_indexer, METH_VARARGS | METH_KEYWORDS, PyDoc_STR("Get an indexer handle")},
        {"index", (PyCFunction)(void*)ffbidx_index, METH_VARARGS | METH_KEYWORDS, PyDoc_STR("Call indexer")},
        {"release", (PyCFunction)(void*)ffbidx_release, METH_VARARGS | METH_KEYWORDS, PyDoc_STR("Release indexer handle")},
        {NULL, NULL, 0, NULL}
    };

    PyModuleDef ffbidx_module = {
        PyModuleDef_HEAD_INIT,
        .m_name = "ffbidx",
        .m_doc = PyDoc_STR("Fast feedback indexer"),
        .m_size = -1,
        .m_methods = ffbidx_methods,
        .m_slots = nullptr,
        .m_traverse = nullptr,
        .m_clear = nullptr,
        .m_free = ffbidx_free
    };

    PyMODINIT_FUNC PyInit_ffbidx(void)
    {
        import_array();
        if (PyErr_Occurred())
            return nullptr;
        PyObject *m = PyModule_Create(&ffbidx_module);
        return m;
    }
    
} // extern "C"
