#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include "mtp_cext.h"
#include "cext_helpers.h"

/* ============================================================================
 * Python wrapper for calc_run
 * Accepts mtp_data as a dict-like object and extracts parameters
 * ============================================================================ */
static PyObject *py_calc_run(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyArrayObject *js_arr, *rs_arr, *itypes_arr, *jtypes_arr;
    PyObject *mtp_data_obj, *mbd_obj;

    /* Parse arguments */
    static char *kwlist[] = {"js", "rs", "itypes", "jtypes", "mtp_data", "mbd", NULL};

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "O!O!O!O!OO",
            kwlist,
            &PyArray_Type, &js_arr,
            &PyArray_Type, &rs_arr,
            &PyArray_Type, &itypes_arr,
            &PyArray_Type, &jtypes_arr,
            &mtp_data_obj,
            &mbd_obj))
    {
        return NULL;
    }

    /* Extract parameters from mtp_data object */
    PyObject *scaling_obj = PyObject_GetAttrString(mtp_data_obj, "scaling");
    PyObject *radial_basis_obj = PyObject_GetAttrString(mtp_data_obj, "radial_basis");
    PyObject *min_dist_obj = NULL;
    PyObject *max_dist_obj = NULL;
    
    if (radial_basis_obj)
    {
        min_dist_obj = PyObject_GetAttrString(radial_basis_obj, "min");
        max_dist_obj = PyObject_GetAttrString(radial_basis_obj, "max");
        Py_DECREF(radial_basis_obj);
    }
    
    PyObject *radial_coeffs_obj = PyObject_GetAttrString(mtp_data_obj, "radial_coeffs");
    PyObject *species_coeffs_obj = PyObject_GetAttrString(mtp_data_obj, "species_coeffs");
    PyObject *species_count_obj = PyObject_GetAttrString(mtp_data_obj, "species_count");
    PyObject *radial_funcs_count_obj = PyObject_GetAttrString(mtp_data_obj, "radial_funcs_count");
    PyObject *alpha_moments_count_obj = PyObject_GetAttrString(mtp_data_obj, "alpha_moments_count");
    PyObject *alpha_moment_mapping_obj = PyObject_GetAttrString(mtp_data_obj, "alpha_moment_mapping");
    PyObject *alpha_index_basic_obj = PyObject_GetAttrString(mtp_data_obj, "alpha_index_basic");
    PyObject *alpha_index_basic_count_obj = PyObject_GetAttrString(mtp_data_obj, "alpha_index_basic_count");
    PyObject *alpha_index_times_obj = PyObject_GetAttrString(mtp_data_obj, "alpha_index_times");
    PyObject *alpha_index_times_count_obj = PyObject_GetAttrString(mtp_data_obj, "alpha_index_times_count");
    PyObject *moment_coeffs_obj = PyObject_GetAttrString(mtp_data_obj, "moment_coeffs");

    if (!scaling_obj || !min_dist_obj || !max_dist_obj || !radial_coeffs_obj ||
        !species_coeffs_obj || !species_count_obj || !radial_funcs_count_obj ||
        !alpha_moments_count_obj || !alpha_moment_mapping_obj || !alpha_index_basic_obj ||
        !alpha_index_basic_count_obj || !alpha_index_times_obj || !alpha_index_times_count_obj ||
        !moment_coeffs_obj)
    {
        return NULL;
    }

    /* Convert to C types */
    double scaling = PyFloat_AsDouble(scaling_obj);
    double min_dist = PyFloat_AsDouble(min_dist_obj);
    double max_dist = PyFloat_AsDouble(max_dist_obj);
    int species_count = (int)PyLong_AsLong(species_count_obj);
    int radial_funcs_count = (int)PyLong_AsLong(radial_funcs_count_obj);
    int alpha_moments_count = (int)PyLong_AsLong(alpha_moments_count_obj);
    int n_basic = (int)PyLong_AsLong(alpha_index_basic_count_obj);
    int n_times = (int)PyLong_AsLong(alpha_index_times_count_obj);

    /* Convert array objects */
    PyArrayObject *radial_coeffs_arr = (PyArrayObject *)PyArray_FROM_OTF(radial_coeffs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *species_coeffs_arr = (PyArrayObject *)PyArray_FROM_OTF(species_coeffs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *alpha_moment_mapping_arr = (PyArrayObject *)PyArray_FROM_OTF(alpha_moment_mapping_obj, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *alpha_index_basic_arr = (PyArrayObject *)PyArray_FROM_OTF(alpha_index_basic_obj, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *alpha_index_times_arr = (PyArrayObject *)PyArray_FROM_OTF(alpha_index_times_obj, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *moment_coeffs_arr = (PyArrayObject *)PyArray_FROM_OTF(moment_coeffs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!radial_coeffs_arr || !species_coeffs_arr || !alpha_moment_mapping_arr ||
        !alpha_index_basic_arr || !alpha_index_times_arr || !moment_coeffs_arr)
    {
        Py_XDECREF(radial_coeffs_arr);
        Py_XDECREF(species_coeffs_arr);
        Py_XDECREF(alpha_moment_mapping_arr);
        Py_XDECREF(alpha_index_basic_arr);
        Py_XDECREF(alpha_index_times_arr);
        Py_XDECREF(moment_coeffs_arr);
        return NULL;
    }

    if (!require_int32(itypes_arr, "itypes") ||
        !require_int32(jtypes_arr, "jtypes") ||
        !require_int32(alpha_moment_mapping_arr, "alpha_moment_mapping") ||
        !require_int32(alpha_index_basic_arr, "alpha_index_basic") ||
        !require_int32(alpha_index_times_arr, "alpha_index_times"))
    {
        Py_DECREF(radial_coeffs_arr);
        Py_DECREF(species_coeffs_arr);
        Py_DECREF(alpha_moment_mapping_arr);
        Py_DECREF(alpha_index_basic_arr);
        Py_DECREF(alpha_index_times_arr);
        Py_DECREF(moment_coeffs_arr);
        return NULL;
    }

    /* Extract array dimensions */
    int n_atoms = (int)PyArray_DIM(rs_arr, 0);
    int n_neighbors = (int)PyArray_DIM(rs_arr, 1);
    int radial_basis_size = (int)PyArray_DIM(radial_coeffs_arr, 3);
    int n_alpha_scalar = (int)PyArray_DIM(alpha_moment_mapping_arr, 0);

    /* Extract mbd.vatoms for in-place writing */
    PyObject *mbd_vatoms_obj = PyObject_GetAttrString(mbd_obj, "vatoms");
    if (!mbd_vatoms_obj)
    {
        Py_DECREF(radial_coeffs_arr);
        Py_DECREF(species_coeffs_arr);
        Py_DECREF(alpha_moment_mapping_arr);
        Py_DECREF(alpha_index_basic_arr);
        Py_DECREF(alpha_index_times_arr);
        Py_DECREF(moment_coeffs_arr);
        return NULL;
    }
    PyArrayObject *mbd_vatoms_arr = get_inplace_double(mbd_vatoms_obj, "mbd.vatoms");
    Py_DECREF(mbd_vatoms_obj);
    if (!mbd_vatoms_arr)
    {
        Py_DECREF(radial_coeffs_arr);
        Py_DECREF(species_coeffs_arr);
        Py_DECREF(alpha_moment_mapping_arr);
        Py_DECREF(alpha_index_basic_arr);
        Py_DECREF(alpha_index_times_arr);
        Py_DECREF(moment_coeffs_arr);
        return NULL;
    }

    /* Create output arrays */
    npy_intp dims_energy[1] = {n_atoms};
    npy_intp dims_grad[3] = {n_atoms, n_neighbors, 3};
    PyArrayObject *energies_arr = (PyArrayObject *)PyArray_ZEROS(1, dims_energy, NPY_DOUBLE, 0);
    PyArrayObject *gradient_arr = (PyArrayObject *)PyArray_ZEROS(3, dims_grad, NPY_DOUBLE, 0);

    if (!energies_arr || !gradient_arr)
    {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate output arrays");
        Py_XDECREF(energies_arr);
        Py_XDECREF(gradient_arr);
        Py_DECREF(radial_coeffs_arr);
        Py_DECREF(species_coeffs_arr);
        Py_DECREF(alpha_moment_mapping_arr);
        Py_DECREF(alpha_index_basic_arr);
        Py_DECREF(alpha_index_times_arr);
        Py_DECREF(moment_coeffs_arr);
        return NULL;
    }

    /* Call C function */
    calc_run(
        n_atoms, n_neighbors,
        (double *)PyArray_DATA(rs_arr),
        (int *)PyArray_DATA(itypes_arr),
        (int *)PyArray_DATA(jtypes_arr),
        scaling, min_dist, max_dist,
        radial_basis_size,
        (double *)PyArray_DATA(radial_coeffs_arr),
        species_count, radial_funcs_count,
        (double *)PyArray_DATA(species_coeffs_arr),
        alpha_moments_count,
        (int *)PyArray_DATA(alpha_moment_mapping_arr),
        n_alpha_scalar,
        (int *)PyArray_DATA(alpha_index_basic_arr),
        n_basic,
        (int *)PyArray_DATA(alpha_index_times_arr),
        n_times,
        (double *)PyArray_DATA(moment_coeffs_arr),
        (double *)PyArray_DATA(energies_arr),
        (double *)PyArray_DATA(gradient_arr),
        (double *)PyArray_DATA(mbd_vatoms_arr));

    /* Cleanup */
    Py_DECREF(radial_coeffs_arr);
    Py_DECREF(species_coeffs_arr);
    Py_DECREF(alpha_moment_mapping_arr);
    Py_DECREF(alpha_index_basic_arr);
    Py_DECREF(alpha_index_times_arr);
    Py_DECREF(moment_coeffs_arr);
    Py_DECREF(scaling_obj);
    Py_DECREF(min_dist_obj);
    Py_DECREF(max_dist_obj);
    Py_DECREF(species_count_obj);
    Py_DECREF(radial_funcs_count_obj);
    Py_DECREF(alpha_moments_count_obj);
    Py_DECREF(alpha_moment_mapping_obj);
    Py_DECREF(alpha_index_basic_obj);
    Py_DECREF(alpha_index_basic_count_obj);
    Py_DECREF(alpha_index_times_obj);
    Py_DECREF(alpha_index_times_count_obj);
    Py_DECREF(moment_coeffs_obj);

    return Py_BuildValue("(OO)", energies_arr, gradient_arr);
}

/* ============================================================================
 * Python wrapper for calc_forces_from_gradient
 * ============================================================================ */
static PyObject *py_calc_forces_from_gradient(PyObject *self, PyObject *args)
{
    PyArrayObject *gradient_arr, *js_arr;

    if (!PyArg_ParseTuple(args, "O!O!",
                          &PyArray_Type, &gradient_arr,
                          &PyArray_Type, &js_arr))
    {
        return NULL;
    }

    int n_atoms = (int)PyArray_DIM(gradient_arr, 0);
    int n_neighbors = (int)PyArray_DIM(gradient_arr, 1);

    /* Create output array */
    npy_intp dims_forces[2] = {n_atoms, 3};
    PyArrayObject *forces_arr = (PyArrayObject *)PyArray_ZEROS(2, dims_forces, NPY_DOUBLE, 0);

    if (!forces_arr)
    {
        return NULL;
    }

    /* Call C function */
    calc_forces_from_gradient(
        (double *)PyArray_DATA(gradient_arr),
        (int *)PyArray_DATA(js_arr),
        n_atoms, n_neighbors,
        (double *)PyArray_DATA(forces_arr));

    return (PyObject *)forces_arr;
}

/* ============================================================================
 * Python wrapper for calc_train (training mode with derivatives)
 * ============================================================================ */
static PyObject *py_calc_train(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyArrayObject *js_arr, *rs_arr, *itypes_arr, *jtypes_arr;
    PyObject *mtp_data_obj;
    PyObject *rbd_obj, *mbd_obj;

    static char *kwlist[] = {"js", "rs", "itypes", "jtypes", "mtp_data", "rbd", "mbd", NULL};

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "O!O!O!O!OOO",
            kwlist,
            &PyArray_Type, &js_arr,
            &PyArray_Type, &rs_arr,
            &PyArray_Type, &itypes_arr,
            &PyArray_Type, &jtypes_arr,
            &mtp_data_obj,
            &rbd_obj,
            &mbd_obj))
    {
        return NULL;
    }

    /* Extract parameters from mtp_data (same as calc_run) */
    PyObject *scaling_obj = PyObject_GetAttrString(mtp_data_obj, "scaling");
    PyObject *radial_basis_obj = PyObject_GetAttrString(mtp_data_obj, "radial_basis");
    PyObject *min_dist_obj = NULL;
    PyObject *max_dist_obj = NULL;
    
    if (radial_basis_obj)
    {
        min_dist_obj = PyObject_GetAttrString(radial_basis_obj, "min");
        max_dist_obj = PyObject_GetAttrString(radial_basis_obj, "max");
        Py_DECREF(radial_basis_obj);
    }
    
    PyObject *radial_coeffs_obj = PyObject_GetAttrString(mtp_data_obj, "radial_coeffs");
    PyObject *species_coeffs_obj = PyObject_GetAttrString(mtp_data_obj, "species_coeffs");
    PyObject *species_count_obj = PyObject_GetAttrString(mtp_data_obj, "species_count");
    PyObject *radial_funcs_count_obj = PyObject_GetAttrString(mtp_data_obj, "radial_funcs_count");
    PyObject *alpha_moments_count_obj = PyObject_GetAttrString(mtp_data_obj, "alpha_moments_count");
    PyObject *alpha_moment_mapping_obj = PyObject_GetAttrString(mtp_data_obj, "alpha_moment_mapping");
    PyObject *alpha_index_basic_obj = PyObject_GetAttrString(mtp_data_obj, "alpha_index_basic");
    PyObject *alpha_index_basic_count_obj = PyObject_GetAttrString(mtp_data_obj, "alpha_index_basic_count");
    PyObject *alpha_index_times_obj = PyObject_GetAttrString(mtp_data_obj, "alpha_index_times");
    PyObject *alpha_index_times_count_obj = PyObject_GetAttrString(mtp_data_obj, "alpha_index_times_count");
    PyObject *moment_coeffs_obj = PyObject_GetAttrString(mtp_data_obj, "moment_coeffs");
    PyObject *alpha_scalar_moments_obj = PyObject_GetAttrString(mtp_data_obj, "alpha_scalar_moments");

    if (!scaling_obj || !min_dist_obj || !max_dist_obj || !radial_coeffs_obj ||
        !species_coeffs_obj || !species_count_obj || !radial_funcs_count_obj ||
        !alpha_moments_count_obj || !alpha_moment_mapping_obj || !alpha_index_basic_obj ||
        !alpha_index_basic_count_obj || !alpha_index_times_obj || !alpha_index_times_count_obj ||
        !moment_coeffs_obj || !alpha_scalar_moments_obj)
    {
        return NULL;
    }

    double scaling = PyFloat_AsDouble(scaling_obj);
    double min_dist = PyFloat_AsDouble(min_dist_obj);
    double max_dist = PyFloat_AsDouble(max_dist_obj);
    int species_count = (int)PyLong_AsLong(species_count_obj);
    int radial_funcs_count = (int)PyLong_AsLong(radial_funcs_count_obj);
    int alpha_moments_count = (int)PyLong_AsLong(alpha_moments_count_obj);
    int n_basic = (int)PyLong_AsLong(alpha_index_basic_count_obj);
    int n_times = (int)PyLong_AsLong(alpha_index_times_count_obj);
    int n_alpha_scalar = (int)PyLong_AsLong(alpha_scalar_moments_obj);

    PyArrayObject *radial_coeffs_arr = (PyArrayObject *)PyArray_FROM_OTF(radial_coeffs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *species_coeffs_arr = (PyArrayObject *)PyArray_FROM_OTF(species_coeffs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *alpha_moment_mapping_arr = (PyArrayObject *)PyArray_FROM_OTF(alpha_moment_mapping_obj, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *alpha_index_basic_arr = (PyArrayObject *)PyArray_FROM_OTF(alpha_index_basic_obj, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *alpha_index_times_arr = (PyArrayObject *)PyArray_FROM_OTF(alpha_index_times_obj, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *moment_coeffs_arr = (PyArrayObject *)PyArray_FROM_OTF(moment_coeffs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!radial_coeffs_arr || !species_coeffs_arr || !alpha_moment_mapping_arr ||
        !alpha_index_basic_arr || !alpha_index_times_arr || !moment_coeffs_arr)
    {
        Py_XDECREF(radial_coeffs_arr);
        Py_XDECREF(species_coeffs_arr);
        Py_XDECREF(alpha_moment_mapping_arr);
        Py_XDECREF(alpha_index_basic_arr);
        Py_XDECREF(alpha_index_times_arr);
        Py_XDECREF(moment_coeffs_arr);
        return NULL;
    }

    if (!require_int32(itypes_arr, "itypes") ||
        !require_int32(jtypes_arr, "jtypes") ||
        !require_int32(alpha_moment_mapping_arr, "alpha_moment_mapping") ||
        !require_int32(alpha_index_basic_arr, "alpha_index_basic") ||
        !require_int32(alpha_index_times_arr, "alpha_index_times"))
    {
        Py_DECREF(radial_coeffs_arr);
        Py_DECREF(species_coeffs_arr);
        Py_DECREF(alpha_moment_mapping_arr);
        Py_DECREF(alpha_index_basic_arr);
        Py_DECREF(alpha_index_times_arr);
        Py_DECREF(moment_coeffs_arr);
        return NULL;
    }

    /* Extract array dimensions */
    int n_atoms = (int)PyArray_DIM(rs_arr, 0);
    int n_neighbors = (int)PyArray_DIM(rs_arr, 1);
    int radial_basis_size = (int)PyArray_DIM(radial_coeffs_arr, 3);

    /* Create energy output array */
    npy_intp dims_energies[1] = {n_atoms};
    PyArrayObject *energies_arr = (PyArrayObject *)PyArray_ZEROS(1, dims_energies, NPY_DOUBLE, 0);

    if (!energies_arr)
    {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate output arrays");
        Py_DECREF(radial_coeffs_arr);
        Py_DECREF(species_coeffs_arr);
        Py_DECREF(alpha_moment_mapping_arr);
        Py_DECREF(alpha_index_basic_arr);
        Py_DECREF(alpha_index_times_arr);
        Py_DECREF(moment_coeffs_arr);
        return NULL;
    }

    /* Extract data pointers from rbd and mbd Python objects */
    PyObject *rbd_values_obj = PyObject_GetAttrString(rbd_obj, "values");
    PyObject *rbd_dqdris_obj = PyObject_GetAttrString(rbd_obj, "dqdris");
    PyObject *rbd_dqdeps_obj = PyObject_GetAttrString(rbd_obj, "dqdeps");
    PyObject *mbd_vatoms_obj = PyObject_GetAttrString(mbd_obj, "vatoms");
    PyObject *mbd_dbdris_obj = PyObject_GetAttrString(mbd_obj, "dbdris");
    PyObject *mbd_dbdeps_obj = PyObject_GetAttrString(mbd_obj, "dbdeps");
    PyObject *mbd_dvdcs_obj = PyObject_GetAttrString(mbd_obj, "dvdcs");
    PyObject *mbd_dgdcs_obj = PyObject_GetAttrString(mbd_obj, "dgdcs");
    PyObject *mbd_dsdcs_obj = PyObject_GetAttrString(mbd_obj, "dsdcs");

    if (!rbd_values_obj || !rbd_dqdris_obj || !rbd_dqdeps_obj ||
        !mbd_vatoms_obj || !mbd_dbdris_obj || !mbd_dbdeps_obj ||
        !mbd_dvdcs_obj || !mbd_dgdcs_obj || !mbd_dsdcs_obj)
    {
        Py_DECREF(radial_coeffs_arr);
        Py_DECREF(species_coeffs_arr);
        Py_DECREF(alpha_moment_mapping_arr);
        Py_DECREF(alpha_index_basic_arr);
        Py_DECREF(alpha_index_times_arr);
        Py_DECREF(moment_coeffs_arr);
        Py_DECREF(energies_arr);
        return NULL;
    }

    PyArrayObject *rbd_values_arr = get_inplace_double(rbd_values_obj, "rbd.values");
    Py_DECREF(rbd_values_obj);
    PyArrayObject *rbd_dqdris_arr = get_inplace_double(rbd_dqdris_obj, "rbd.dqdris");
    Py_DECREF(rbd_dqdris_obj);
    PyArrayObject *rbd_dqdeps_arr = get_inplace_double(rbd_dqdeps_obj, "rbd.dqdeps");
    Py_DECREF(rbd_dqdeps_obj);
    PyArrayObject *mbd_vatoms_arr = get_inplace_double(mbd_vatoms_obj, "mbd.vatoms");
    Py_DECREF(mbd_vatoms_obj);
    PyArrayObject *mbd_dbdris_arr = get_inplace_double(mbd_dbdris_obj, "mbd.dbdris");
    Py_DECREF(mbd_dbdris_obj);
    PyArrayObject *mbd_dbdeps_arr = get_inplace_double(mbd_dbdeps_obj, "mbd.dbdeps");
    Py_DECREF(mbd_dbdeps_obj);
    PyArrayObject *mbd_dvdcs_arr = get_inplace_double(mbd_dvdcs_obj, "mbd.dvdcs");
    Py_DECREF(mbd_dvdcs_obj);
    PyArrayObject *mbd_dgdcs_arr = get_inplace_double(mbd_dgdcs_obj, "mbd.dgdcs");
    Py_DECREF(mbd_dgdcs_obj);
    PyArrayObject *mbd_dsdcs_arr = get_inplace_double(mbd_dsdcs_obj, "mbd.dsdcs");
    Py_DECREF(mbd_dsdcs_obj);

    if (!rbd_values_arr || !rbd_dqdris_arr || !rbd_dqdeps_arr ||
        !mbd_vatoms_arr || !mbd_dbdris_arr || !mbd_dbdeps_arr ||
        !mbd_dvdcs_arr || !mbd_dgdcs_arr || !mbd_dsdcs_arr)
    {
        Py_DECREF(radial_coeffs_arr);
        Py_DECREF(species_coeffs_arr);
        Py_DECREF(alpha_moment_mapping_arr);
        Py_DECREF(alpha_index_basic_arr);
        Py_DECREF(alpha_index_times_arr);
        Py_DECREF(moment_coeffs_arr);
        Py_DECREF(energies_arr);
        return NULL;
    }

    /* Create C struct wrappers for data pointers */
    RadialBasisData rbd = {
        .values = (double *)PyArray_DATA(rbd_values_arr),
        .dqdris = (double *)PyArray_DATA(rbd_dqdris_arr),
        .dqdeps = (double *)PyArray_DATA(rbd_dqdeps_arr)};

    MomentBasisData mbd = {
        .vatoms = (double *)PyArray_DATA(mbd_vatoms_arr),
        .dbdris = (double *)PyArray_DATA(mbd_dbdris_arr),
        .dbdeps = (double *)PyArray_DATA(mbd_dbdeps_arr),
        .dvdcs = (double *)PyArray_DATA(mbd_dvdcs_arr),
        .dgdcs = (double *)PyArray_DATA(mbd_dgdcs_arr),
        .dsdcs = (double *)PyArray_DATA(mbd_dsdcs_arr)};

    /* Call C function with struct pointers */
    calc_train(
        n_atoms, n_neighbors,
        (double *)PyArray_DATA(rs_arr),
        (int *)PyArray_DATA(js_arr),
        (int *)PyArray_DATA(itypes_arr),
        (int *)PyArray_DATA(jtypes_arr),
        scaling, min_dist, max_dist,
        radial_basis_size,
        (double *)PyArray_DATA(radial_coeffs_arr),
        species_count, radial_funcs_count,
        (double *)PyArray_DATA(species_coeffs_arr),
        alpha_moments_count,
        (int *)PyArray_DATA(alpha_moment_mapping_arr),
        n_alpha_scalar,
        (int *)PyArray_DATA(alpha_index_basic_arr),
        n_basic,
        (int *)PyArray_DATA(alpha_index_times_arr),
        n_times,
        (double *)PyArray_DATA(moment_coeffs_arr),
        (double *)PyArray_DATA(energies_arr),
        &rbd,
        &mbd);

    /* Cleanup */
    Py_DECREF(radial_coeffs_arr);
    Py_DECREF(species_coeffs_arr);
    Py_DECREF(alpha_moment_mapping_arr);
    Py_DECREF(alpha_index_basic_arr);
    Py_DECREF(alpha_index_times_arr);
    Py_DECREF(moment_coeffs_arr);
    Py_DECREF(scaling_obj);
    Py_DECREF(min_dist_obj);
    Py_DECREF(max_dist_obj);
    Py_DECREF(species_count_obj);
    Py_DECREF(radial_funcs_count_obj);
    Py_DECREF(alpha_moments_count_obj);
    Py_DECREF(alpha_moment_mapping_obj);
    Py_DECREF(alpha_index_basic_obj);
    Py_DECREF(alpha_index_basic_count_obj);
    Py_DECREF(alpha_index_times_obj);
    Py_DECREF(alpha_index_times_count_obj);
    Py_DECREF(moment_coeffs_obj);
    Py_DECREF(alpha_scalar_moments_obj);

    return (PyObject *)energies_arr;
}

static PyMethodDef MtpCextMethods[] = {
    {"calc_run", (PyCFunction)py_calc_run, METH_VARARGS | METH_KEYWORDS,
     "Calculate run mode with mtp_data object.\n"
     "calc_run(js, rs, itypes, jtypes, mtp_data, mbd)\n"
     "Writes moment basis values per atom in-place into mbd.vatoms.\n"
     "Returns (energies, gradient)"},
    {"calc_train", (PyCFunction)py_calc_train, METH_VARARGS | METH_KEYWORDS,
     "Calculate training mode with parameter derivatives.\n"
     "calc_train(js, rs, itypes, jtypes, mtp_data, rbd, mbd)\n"
     "Writes results directly into preallocated rbd/mbd arrays.\n"
     "Returns energies array"},
    {"calc_forces_from_gradient", py_calc_forces_from_gradient, METH_VARARGS,
     "Calculate forces from gradients.\n"
     "calc_forces_from_gradient(gradient, js)\n"
     "Returns forces"},
    {NULL, NULL, 0, NULL}};

/* Module definition */
static struct PyModuleDef mtpcextmodule = {
    PyModuleDef_HEAD_INIT,
    "_mtp_cext",
    "C extension for MTP calculations (run mode only)",
    -1,
    MtpCextMethods};

/* Module initialization */
PyMODINIT_FUNC PyInit__mtp_cext(void)
{
    import_array();
    return PyModule_Create(&mtpcextmodule);
}
