#pragma once
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

static int require_int32(PyArrayObject *arr, const char *name)
{
    if (PyArray_TYPE(arr) != NPY_INT32)
    {
        PyErr_Format(PyExc_TypeError, "%s must be int32 array", name);
        return 0;
    }
    return 1;
}

static int require_inplace_double(PyArrayObject *arr, const char *name)
{
    if (PyArray_TYPE(arr) != NPY_DOUBLE)
    {
        PyErr_Format(PyExc_TypeError, "%s must be float64 array", name);
        return 0;
    }
    if (!(PyArray_FLAGS(arr) & NPY_ARRAY_WRITEABLE))
    {
        PyErr_Format(PyExc_ValueError, "%s must be writable", name);
        return 0;
    }
    if (!(PyArray_FLAGS(arr) & NPY_ARRAY_C_CONTIGUOUS))
    {
        PyErr_Format(PyExc_ValueError, "%s must be C-contiguous", name);
        return 0;
    }
    if (!PyArray_ISNOTSWAPPED(arr))
    {
        PyErr_Format(PyExc_ValueError, "%s must be native byte order", name);
        return 0;
    }
    if (!(PyArray_FLAGS(arr) & NPY_ARRAY_ALIGNED))
    {
        PyErr_Format(PyExc_ValueError, "%s must be aligned", name);
        return 0;
    }
    if (PyArray_FLAGS(arr) & NPY_ARRAY_WRITEBACKIFCOPY)
    {
        PyErr_Format(PyExc_ValueError, "%s must be in-place (no writeback copy)", name);
        return 0;
    }
    return 1;
}

static PyArrayObject *get_inplace_double(PyObject *obj, const char *name)
{
    if (!PyArray_Check(obj))
    {
        PyErr_Format(PyExc_TypeError, "%s must be a numpy array", name);
        return NULL;
    }
    PyArrayObject *arr = (PyArrayObject *)obj;
    if (!require_inplace_double(arr, name))
        return NULL;
    return arr;
}
