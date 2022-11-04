#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <iostream>
#include <sstream>
#include <limits>
#include <atomic>
#include <map>
#include <exception>
#include "indexer.h"

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

        constexpr const char* kw[] = {"handle", "length_threshold", "num_sample_points", "n_cells", "data", nullptr};
        double length_threshold;
        long handle, num_sample_points, n_cells;
        PyArrayObject* ndarray;
        if (PyArg_ParseTupleAndKeywords(args, kwds, "ldllO!", (char**)kw, &handle, &length_threshold, &num_sample_points, &n_cells, &PyArray_Type, &ndarray) == 0)
            return nullptr;
        
        int ndim = PyArray_NDIM(ndarray);
        std::ostringstream oss;
        oss << "handle=" << handle << " length_threshold=" << length_threshold << " num_sample_points=" << num_sample_points
            << " n_cells=" << n_cells << " ndim=" << ndim << '\n';
        PyErr_SetString(PyExc_RuntimeError, oss.str().c_str());
        return nullptr;
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

        if (indexers.erase(handle) != 1) {
            PyErr_SetString(PyExc_ValueError, "invalid handle");
            return nullptr;
        }

        Py_RETURN_NONE;
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
        {"indexer", (PyCFunction)ffbidx_indexer, METH_VARARGS | METH_KEYWORDS, PyDoc_STR("Get an indexer handle")},
        {"index", (PyCFunction)ffbidx_index, METH_VARARGS | METH_KEYWORDS, PyDoc_STR("Call indexer")},
        {"release", (PyCFunction)ffbidx_release, METH_VARARGS | METH_KEYWORDS, PyDoc_STR("Release indexer handle")},
        {NULL, NULL, 0, NULL}
    };

    PyModuleDef ffbidx_module = {
        PyModuleDef_HEAD_INIT,
        .m_name = "ffbidx",
        .m_doc = PyDoc_STR("Fast feedback indexer"),
        .m_size = -1,
        .m_methods = ffbidx_methods,
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
