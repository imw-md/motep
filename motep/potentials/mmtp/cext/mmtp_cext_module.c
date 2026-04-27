#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include "mmtp_cext.h"
#include "cext_helpers.h"

/* ==========================================================================
 * Python wrapper for calc_mag_run
 * ========================================================================== */
static PyObject *py_calc_mag_run(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyArrayObject *js_arr, *rs_arr, *ms_arr, *itypes_arr, *jtypes_arr;
    PyObject *mtp_data_obj, *mbd_obj;

    static char *kwlist[] = {"js", "rs", "ms", "itypes", "jtypes", "mtp_data", "mbd", NULL};

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "O!O!O!O!O!OO",
            kwlist,
            &PyArray_Type, &js_arr,
            &PyArray_Type, &rs_arr,
            &PyArray_Type, &ms_arr,
            &PyArray_Type, &itypes_arr,
            &PyArray_Type, &jtypes_arr,
            &mtp_data_obj,
            &mbd_obj))
        return NULL;

    /* All owned references, NULL-initialized for safe Py_XDECREF in cleanup */
    PyObject *result = NULL;
    PyObject *scaling_obj = NULL, *min_dist_obj = NULL, *max_dist_obj = NULL;
    PyObject *min_mag_obj = NULL, *max_mag_obj = NULL;
    PyObject *mag_basis_size_obj = NULL, *radial_basis_size_obj = NULL;
    PyObject *radial_coeffs_obj = NULL, *species_coeffs_obj = NULL;
    PyObject *species_count_obj = NULL, *radial_funcs_count_obj = NULL;
    PyObject *alpha_moments_count_obj = NULL, *alpha_moment_mapping_obj = NULL;
    PyObject *alpha_index_basic_obj = NULL, *alpha_index_basic_count_obj = NULL;
    PyObject *alpha_index_times_obj = NULL, *alpha_index_times_count_obj = NULL;
    PyObject *moment_coeffs_obj = NULL;
    PyArrayObject *radial_coeffs_arr = NULL, *species_coeffs_arr = NULL;
    PyArrayObject *alpha_moment_mapping_arr = NULL;
    PyArrayObject *alpha_index_basic_arr = NULL, *alpha_index_times_arr = NULL;
    PyArrayObject *moment_coeffs_arr = NULL;
    PyArrayObject *energies_arr = NULL, *gradient_arr = NULL;
    PyArrayObject *grad_mag_i_arr = NULL, *grad_mag_j_arr = NULL;

    PyObject *radial_basis_obj = PyObject_GetAttrString(mtp_data_obj, "radial_basis");
    if (radial_basis_obj)
    {
        min_dist_obj = PyObject_GetAttrString(radial_basis_obj, "min");
        max_dist_obj = PyObject_GetAttrString(radial_basis_obj, "max");
        radial_basis_size_obj = PyObject_GetAttrString(radial_basis_obj, "size");
        Py_DECREF(radial_basis_obj);
    }
    PyObject *magnetic_basis_obj = PyObject_GetAttrString(mtp_data_obj, "magnetic_basis");
    if (magnetic_basis_obj)
    {
        min_mag_obj = PyObject_GetAttrString(magnetic_basis_obj, "min");
        max_mag_obj = PyObject_GetAttrString(magnetic_basis_obj, "max");
        mag_basis_size_obj = PyObject_GetAttrString(magnetic_basis_obj, "size");
        Py_DECREF(magnetic_basis_obj);
    }

    scaling_obj = PyObject_GetAttrString(mtp_data_obj, "scaling");
    radial_coeffs_obj = PyObject_GetAttrString(mtp_data_obj, "radial_coeffs");
    species_coeffs_obj = PyObject_GetAttrString(mtp_data_obj, "species_coeffs");
    species_count_obj = PyObject_GetAttrString(mtp_data_obj, "species_count");
    radial_funcs_count_obj = PyObject_GetAttrString(mtp_data_obj, "radial_funcs_count");
    alpha_moments_count_obj = PyObject_GetAttrString(mtp_data_obj, "alpha_moments_count");
    alpha_moment_mapping_obj = PyObject_GetAttrString(mtp_data_obj, "alpha_moment_mapping");
    alpha_index_basic_obj = PyObject_GetAttrString(mtp_data_obj, "alpha_index_basic");
    alpha_index_basic_count_obj = PyObject_GetAttrString(mtp_data_obj, "alpha_index_basic_count");
    alpha_index_times_obj = PyObject_GetAttrString(mtp_data_obj, "alpha_index_times");
    alpha_index_times_count_obj = PyObject_GetAttrString(mtp_data_obj, "alpha_index_times_count");
    moment_coeffs_obj = PyObject_GetAttrString(mtp_data_obj, "moment_coeffs");

    if (!scaling_obj || !min_dist_obj || !max_dist_obj || !min_mag_obj || !max_mag_obj ||
        !mag_basis_size_obj || !radial_basis_size_obj || !radial_coeffs_obj ||
        !species_coeffs_obj || !species_count_obj || !radial_funcs_count_obj ||
        !alpha_moments_count_obj || !alpha_moment_mapping_obj || !alpha_index_basic_obj ||
        !alpha_index_basic_count_obj || !alpha_index_times_obj ||
        !alpha_index_times_count_obj || !moment_coeffs_obj)
        goto cleanup;

    double scaling = PyFloat_AsDouble(scaling_obj);
    double min_dist = PyFloat_AsDouble(min_dist_obj);
    double max_dist = PyFloat_AsDouble(max_dist_obj);
    double min_mag = PyFloat_AsDouble(min_mag_obj);
    double max_mag = PyFloat_AsDouble(max_mag_obj);
    int mag_basis_size = (int)PyLong_AsLong(mag_basis_size_obj);
    int species_count = (int)PyLong_AsLong(species_count_obj);
    int radial_funcs_count = (int)PyLong_AsLong(radial_funcs_count_obj);
    int alpha_moments_count = (int)PyLong_AsLong(alpha_moments_count_obj);
    int n_basic = (int)PyLong_AsLong(alpha_index_basic_count_obj);
    int n_times = (int)PyLong_AsLong(alpha_index_times_count_obj);

    radial_coeffs_arr = (PyArrayObject *)PyArray_FROM_OTF(radial_coeffs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    species_coeffs_arr = (PyArrayObject *)PyArray_FROM_OTF(species_coeffs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    alpha_moment_mapping_arr = (PyArrayObject *)PyArray_FROM_OTF(alpha_moment_mapping_obj, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);
    alpha_index_basic_arr = (PyArrayObject *)PyArray_FROM_OTF(alpha_index_basic_obj, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);
    alpha_index_times_arr = (PyArrayObject *)PyArray_FROM_OTF(alpha_index_times_obj, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);
    moment_coeffs_arr = (PyArrayObject *)PyArray_FROM_OTF(moment_coeffs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!radial_coeffs_arr || !species_coeffs_arr || !alpha_moment_mapping_arr ||
        !alpha_index_basic_arr || !alpha_index_times_arr || !moment_coeffs_arr)
        goto cleanup;

    if (!require_int32(js_arr, "js") ||
        !require_int32(itypes_arr, "itypes") ||
        !require_int32(jtypes_arr, "jtypes") ||
        !require_int32(alpha_moment_mapping_arr, "alpha_moment_mapping") ||
        !require_int32(alpha_index_basic_arr, "alpha_index_basic") ||
        !require_int32(alpha_index_times_arr, "alpha_index_times"))
        goto cleanup;

    int n_atoms = (int)PyArray_DIM(rs_arr, 0);
    int n_neighbors = (int)PyArray_DIM(rs_arr, 1);
    int radial_basis_size = (int)PyLong_AsLong(radial_basis_size_obj);
    int n_alpha_scalar = (int)PyArray_DIM(alpha_moment_mapping_arr, 0);

    npy_intp dims_energy[1] = {n_atoms};
    npy_intp dims_grad[3] = {n_atoms, n_neighbors, 3};
    npy_intp dims_grad_mag[2] = {n_atoms, n_neighbors};
    energies_arr = (PyArrayObject *)PyArray_ZEROS(1, dims_energy, NPY_DOUBLE, 0);
    gradient_arr = (PyArrayObject *)PyArray_ZEROS(3, dims_grad, NPY_DOUBLE, 0);
    grad_mag_i_arr = (PyArrayObject *)PyArray_ZEROS(2, dims_grad_mag, NPY_DOUBLE, 0);
    grad_mag_j_arr = (PyArrayObject *)PyArray_ZEROS(2, dims_grad_mag, NPY_DOUBLE, 0);

    if (!energies_arr || !gradient_arr || !grad_mag_i_arr || !grad_mag_j_arr)
    {
        if (!PyErr_Occurred())
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate output arrays");
        goto cleanup;
    }

    /* mbd_vatoms_arr is a borrowed view — obj decref'd inline, arr not owned */
    PyObject *mbd_vatoms_obj = PyObject_GetAttrString(mbd_obj, "vatoms");
    PyArrayObject *mbd_vatoms_arr = mbd_vatoms_obj ? get_inplace_double(mbd_vatoms_obj, "mbd.vatoms") : NULL;
    Py_XDECREF(mbd_vatoms_obj);
    if (!mbd_vatoms_arr)
        goto cleanup;

    calc_mag_run(
        n_atoms, n_neighbors,
        (double *)PyArray_DATA(rs_arr),
        (int *)PyArray_DATA(js_arr),
        (double *)PyArray_DATA(ms_arr),
        (int *)PyArray_DATA(itypes_arr),
        (int *)PyArray_DATA(jtypes_arr),
        scaling, min_dist, max_dist,
        min_mag, max_mag,
        radial_basis_size,
        mag_basis_size,
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
        (double *)PyArray_DATA(grad_mag_i_arr),
        (double *)PyArray_DATA(grad_mag_j_arr),
        (double *)PyArray_DATA(mbd_vatoms_arr));

    /* N steals refs — NULL out so cleanup's Py_XDECREF is a no-op for these */
    result = Py_BuildValue("(NNNN)", energies_arr, gradient_arr, grad_mag_i_arr, grad_mag_j_arr);
    energies_arr = gradient_arr = grad_mag_i_arr = grad_mag_j_arr = NULL;

cleanup:
    Py_XDECREF(energies_arr);
    Py_XDECREF(gradient_arr);
    Py_XDECREF(grad_mag_i_arr);
    Py_XDECREF(grad_mag_j_arr);
    Py_XDECREF(radial_coeffs_arr);
    Py_XDECREF(species_coeffs_arr);
    Py_XDECREF(alpha_moment_mapping_arr);
    Py_XDECREF(alpha_index_basic_arr);
    Py_XDECREF(alpha_index_times_arr);
    Py_XDECREF(moment_coeffs_arr);
    Py_XDECREF(scaling_obj);
    Py_XDECREF(min_dist_obj);
    Py_XDECREF(max_dist_obj);
    Py_XDECREF(min_mag_obj);
    Py_XDECREF(max_mag_obj);
    Py_XDECREF(mag_basis_size_obj);
    Py_XDECREF(radial_basis_size_obj);
    Py_XDECREF(radial_coeffs_obj);
    Py_XDECREF(species_coeffs_obj);
    Py_XDECREF(species_count_obj);
    Py_XDECREF(radial_funcs_count_obj);
    Py_XDECREF(alpha_moments_count_obj);
    Py_XDECREF(alpha_moment_mapping_obj);
    Py_XDECREF(alpha_index_basic_obj);
    Py_XDECREF(alpha_index_basic_count_obj);
    Py_XDECREF(alpha_index_times_obj);
    Py_XDECREF(alpha_index_times_count_obj);
    Py_XDECREF(moment_coeffs_obj);
    return result;
}

/* ==========================================================================
 * Python wrapper for calc_forces_from_gradient
 * ========================================================================== */
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

    npy_intp dims_forces[2] = {n_atoms, 3};
    PyArrayObject *forces_arr = (PyArrayObject *)PyArray_ZEROS(2, dims_forces, NPY_DOUBLE, 0);

    if (!forces_arr)
    {
        return NULL;
    }

    calc_forces_from_gradient(
        (double *)PyArray_DATA(gradient_arr),
        (int *)PyArray_DATA(js_arr),
        n_atoms, n_neighbors,
        (double *)PyArray_DATA(forces_arr));

    return (PyObject *)forces_arr;
}

/* ==========================================================================
 * Python wrapper for calc_mgrad_from_gradient
 * ========================================================================== */
static PyObject *py_calc_mgrad_from_gradient(PyObject *self, PyObject *args)
{
    PyArrayObject *grad_mag_i_arr, *grad_mag_j_arr, *js_arr;

    if (!PyArg_ParseTuple(args, "O!O!O!",
                          &PyArray_Type, &grad_mag_i_arr,
                          &PyArray_Type, &grad_mag_j_arr,
                          &PyArray_Type, &js_arr))
    {
        return NULL;
    }

    int n_atoms = (int)PyArray_DIM(grad_mag_i_arr, 0);
    int n_neighbors = (int)PyArray_DIM(grad_mag_i_arr, 1);

    npy_intp dims_mgrad[1] = {n_atoms};
    PyArrayObject *mgrad_arr = (PyArrayObject *)PyArray_ZEROS(1, dims_mgrad, NPY_DOUBLE, 0);

    if (!mgrad_arr)
    {
        return NULL;
    }

    calc_mgrad_from_gradient(
        (double *)PyArray_DATA(grad_mag_i_arr),
        (double *)PyArray_DATA(grad_mag_j_arr),
        (int *)PyArray_DATA(js_arr),
        n_atoms, n_neighbors,
        (double *)PyArray_DATA(mgrad_arr));

    return (PyObject *)mgrad_arr;
}

/* ==========================================================================
 * Python wrapper for calc_mag_train
 * ========================================================================== */
static PyObject *py_calc_mag_train(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyArrayObject *js_arr, *rs_arr, *ms_arr, *itypes_arr, *jtypes_arr;
    PyObject *mtp_data_obj, *rbd_obj, *mbd_obj;

    static char *kwlist[] = {"js", "rs", "ms", "itypes", "jtypes", "mtp_data", "rbd", "mbd", NULL};

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "O!O!O!O!O!OOO",
            kwlist,
            &PyArray_Type, &js_arr,
            &PyArray_Type, &rs_arr,
            &PyArray_Type, &ms_arr,
            &PyArray_Type, &itypes_arr,
            &PyArray_Type, &jtypes_arr,
            &mtp_data_obj,
            &rbd_obj,
            &mbd_obj))
        return NULL;

    /* All owned references, NULL-initialized for safe Py_XDECREF in cleanup */
    PyObject *result = NULL;
    PyObject *scaling_obj = NULL, *min_dist_obj = NULL, *max_dist_obj = NULL;
    PyObject *min_mag_obj = NULL, *max_mag_obj = NULL;
    PyObject *mag_basis_size_obj = NULL, *radial_basis_size_obj = NULL;
    PyObject *radial_coeffs_obj = NULL, *species_coeffs_obj = NULL;
    PyObject *species_count_obj = NULL, *radial_funcs_count_obj = NULL;
    PyObject *alpha_moments_count_obj = NULL, *alpha_moment_mapping_obj = NULL;
    PyObject *alpha_index_basic_obj = NULL, *alpha_index_basic_count_obj = NULL;
    PyObject *alpha_index_times_obj = NULL, *alpha_index_times_count_obj = NULL;
    PyObject *moment_coeffs_obj = NULL;
    PyArrayObject *radial_coeffs_arr = NULL, *species_coeffs_arr = NULL;
    PyArrayObject *alpha_moment_mapping_arr = NULL;
    PyArrayObject *alpha_index_basic_arr = NULL, *alpha_index_times_arr = NULL;
    PyArrayObject *moment_coeffs_arr = NULL;
    PyArrayObject *energies_arr = NULL;

    PyObject *radial_basis_obj = PyObject_GetAttrString(mtp_data_obj, "radial_basis");
    if (radial_basis_obj)
    {
        min_dist_obj = PyObject_GetAttrString(radial_basis_obj, "min");
        max_dist_obj = PyObject_GetAttrString(radial_basis_obj, "max");
        radial_basis_size_obj = PyObject_GetAttrString(radial_basis_obj, "size");
        Py_DECREF(radial_basis_obj);
    }
    PyObject *magnetic_basis_obj = PyObject_GetAttrString(mtp_data_obj, "magnetic_basis");
    if (magnetic_basis_obj)
    {
        min_mag_obj = PyObject_GetAttrString(magnetic_basis_obj, "min");
        max_mag_obj = PyObject_GetAttrString(magnetic_basis_obj, "max");
        mag_basis_size_obj = PyObject_GetAttrString(magnetic_basis_obj, "size");
        Py_DECREF(magnetic_basis_obj);
    }

    scaling_obj = PyObject_GetAttrString(mtp_data_obj, "scaling");
    radial_coeffs_obj = PyObject_GetAttrString(mtp_data_obj, "radial_coeffs");
    species_coeffs_obj = PyObject_GetAttrString(mtp_data_obj, "species_coeffs");
    species_count_obj = PyObject_GetAttrString(mtp_data_obj, "species_count");
    radial_funcs_count_obj = PyObject_GetAttrString(mtp_data_obj, "radial_funcs_count");
    alpha_moments_count_obj = PyObject_GetAttrString(mtp_data_obj, "alpha_moments_count");
    alpha_moment_mapping_obj = PyObject_GetAttrString(mtp_data_obj, "alpha_moment_mapping");
    alpha_index_basic_obj = PyObject_GetAttrString(mtp_data_obj, "alpha_index_basic");
    alpha_index_basic_count_obj = PyObject_GetAttrString(mtp_data_obj, "alpha_index_basic_count");
    alpha_index_times_obj = PyObject_GetAttrString(mtp_data_obj, "alpha_index_times");
    alpha_index_times_count_obj = PyObject_GetAttrString(mtp_data_obj, "alpha_index_times_count");
    moment_coeffs_obj = PyObject_GetAttrString(mtp_data_obj, "moment_coeffs");

    if (!scaling_obj || !min_dist_obj || !max_dist_obj || !min_mag_obj || !max_mag_obj ||
        !mag_basis_size_obj || !radial_basis_size_obj || !radial_coeffs_obj || !species_coeffs_obj || !species_count_obj ||
        !radial_funcs_count_obj || !alpha_moments_count_obj || !alpha_moment_mapping_obj ||
        !alpha_index_basic_obj || !alpha_index_basic_count_obj || !alpha_index_times_obj ||
        !alpha_index_times_count_obj || !moment_coeffs_obj)
        goto cleanup;

    double scaling = PyFloat_AsDouble(scaling_obj);
    double min_dist = PyFloat_AsDouble(min_dist_obj);
    double max_dist = PyFloat_AsDouble(max_dist_obj);
    double min_mag = PyFloat_AsDouble(min_mag_obj);
    double max_mag = PyFloat_AsDouble(max_mag_obj);
    int mag_basis_size = (int)PyLong_AsLong(mag_basis_size_obj);
    int species_count = (int)PyLong_AsLong(species_count_obj);
    int radial_funcs_count = (int)PyLong_AsLong(radial_funcs_count_obj);
    int alpha_moments_count = (int)PyLong_AsLong(alpha_moments_count_obj);
    int n_basic = (int)PyLong_AsLong(alpha_index_basic_count_obj);
    int n_times = (int)PyLong_AsLong(alpha_index_times_count_obj);

    radial_coeffs_arr = (PyArrayObject *)PyArray_FROM_OTF(radial_coeffs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    species_coeffs_arr = (PyArrayObject *)PyArray_FROM_OTF(species_coeffs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    alpha_moment_mapping_arr = (PyArrayObject *)PyArray_FROM_OTF(alpha_moment_mapping_obj, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);
    alpha_index_basic_arr = (PyArrayObject *)PyArray_FROM_OTF(alpha_index_basic_obj, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);
    alpha_index_times_arr = (PyArrayObject *)PyArray_FROM_OTF(alpha_index_times_obj, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);
    moment_coeffs_arr = (PyArrayObject *)PyArray_FROM_OTF(moment_coeffs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!radial_coeffs_arr || !species_coeffs_arr || !alpha_moment_mapping_arr ||
        !alpha_index_basic_arr || !alpha_index_times_arr || !moment_coeffs_arr)
        goto cleanup;

    if (!require_int32(js_arr, "js") ||
        !require_int32(itypes_arr, "itypes") ||
        !require_int32(jtypes_arr, "jtypes") ||
        !require_int32(alpha_moment_mapping_arr, "alpha_moment_mapping") ||
        !require_int32(alpha_index_basic_arr, "alpha_index_basic") ||
        !require_int32(alpha_index_times_arr, "alpha_index_times"))
        goto cleanup;

    int n_atoms = (int)PyArray_DIM(rs_arr, 0);
    int n_neighbors = (int)PyArray_DIM(rs_arr, 1);
    int radial_basis_size = (int)PyLong_AsLong(radial_basis_size_obj);
    int n_alpha_scalar = (int)PyArray_DIM(alpha_moment_mapping_arr, 0);

    /* rbd/mbd borrowed views — obj decref'd inline, arr not owned */
    PyObject *rbd_values_obj = PyObject_GetAttrString(rbd_obj, "values");
    PyArrayObject *rbd_values_arr = rbd_values_obj ? get_inplace_double(rbd_values_obj, "rbd.values") : NULL;
    Py_XDECREF(rbd_values_obj);
    PyObject *rbd_dqdris_obj = PyObject_GetAttrString(rbd_obj, "dqdris");
    PyArrayObject *rbd_dqdris_arr = rbd_dqdris_obj ? get_inplace_double(rbd_dqdris_obj, "rbd.dqdris") : NULL;
    Py_XDECREF(rbd_dqdris_obj);
    PyObject *rbd_dqdeps_obj = PyObject_GetAttrString(rbd_obj, "dqdeps");
    PyArrayObject *rbd_dqdeps_arr = rbd_dqdeps_obj ? get_inplace_double(rbd_dqdeps_obj, "rbd.dqdeps") : NULL;
    Py_XDECREF(rbd_dqdeps_obj);
    PyObject *mbd_vatoms_obj = PyObject_GetAttrString(mbd_obj, "vatoms");
    PyArrayObject *mbd_vatoms_arr = mbd_vatoms_obj ? get_inplace_double(mbd_vatoms_obj, "mbd.vatoms") : NULL;
    Py_XDECREF(mbd_vatoms_obj);
    PyObject *mbd_dbdris_obj = PyObject_GetAttrString(mbd_obj, "dbdris");
    PyArrayObject *mbd_dbdris_arr = mbd_dbdris_obj ? get_inplace_double(mbd_dbdris_obj, "mbd.dbdris") : NULL;
    Py_XDECREF(mbd_dbdris_obj);
    PyObject *mbd_dbdeps_obj = PyObject_GetAttrString(mbd_obj, "dbdeps");
    PyArrayObject *mbd_dbdeps_arr = mbd_dbdeps_obj ? get_inplace_double(mbd_dbdeps_obj, "mbd.dbdeps") : NULL;
    Py_XDECREF(mbd_dbdeps_obj);
    PyObject *mbd_dvdcs_obj = PyObject_GetAttrString(mbd_obj, "dvdcs");
    PyArrayObject *mbd_dvdcs_arr = mbd_dvdcs_obj ? get_inplace_double(mbd_dvdcs_obj, "mbd.dvdcs") : NULL;
    Py_XDECREF(mbd_dvdcs_obj);
    PyObject *mbd_dgdcs_obj = PyObject_GetAttrString(mbd_obj, "dgdcs");
    PyArrayObject *mbd_dgdcs_arr = mbd_dgdcs_obj ? get_inplace_double(mbd_dgdcs_obj, "mbd.dgdcs") : NULL;
    Py_XDECREF(mbd_dgdcs_obj);
    PyObject *mbd_dsdcs_obj = PyObject_GetAttrString(mbd_obj, "dsdcs");
    PyArrayObject *mbd_dsdcs_arr = mbd_dsdcs_obj ? get_inplace_double(mbd_dsdcs_obj, "mbd.dsdcs") : NULL;
    Py_XDECREF(mbd_dsdcs_obj);

    if (!rbd_values_arr || !rbd_dqdris_arr || !rbd_dqdeps_arr ||
        !mbd_vatoms_arr || !mbd_dbdris_arr || !mbd_dbdeps_arr ||
        !mbd_dvdcs_arr || !mbd_dgdcs_arr || !mbd_dsdcs_arr)
        goto cleanup;

    npy_intp dims_energy[1] = {n_atoms};
    energies_arr = (PyArrayObject *)PyArray_ZEROS(1, dims_energy, NPY_DOUBLE, 0);

    if (!energies_arr)
    {
        if (!PyErr_Occurred())
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate output arrays");
        goto cleanup;
    }

    calc_mag_train(
        n_atoms, n_neighbors,
        (double *)PyArray_DATA(rs_arr),
        (int *)PyArray_DATA(js_arr),
        (double *)PyArray_DATA(ms_arr),
        (int *)PyArray_DATA(itypes_arr),
        (int *)PyArray_DATA(jtypes_arr),
        scaling, min_dist, max_dist,
        min_mag, max_mag,
        radial_basis_size,
        mag_basis_size,
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
        (double *)PyArray_DATA(rbd_values_arr),
        (double *)PyArray_DATA(rbd_dqdris_arr),
        (double *)PyArray_DATA(rbd_dqdeps_arr),
        (double *)PyArray_DATA(mbd_vatoms_arr),
        (double *)PyArray_DATA(mbd_dbdris_arr),
        (double *)PyArray_DATA(mbd_dbdeps_arr),
        (double *)PyArray_DATA(mbd_dvdcs_arr),
        (double *)PyArray_DATA(mbd_dgdcs_arr),
        (double *)PyArray_DATA(mbd_dsdcs_arr));

    result = (PyObject *)energies_arr;
    energies_arr = NULL;

cleanup:
    Py_XDECREF(energies_arr);
    Py_XDECREF(radial_coeffs_arr);
    Py_XDECREF(species_coeffs_arr);
    Py_XDECREF(alpha_moment_mapping_arr);
    Py_XDECREF(alpha_index_basic_arr);
    Py_XDECREF(alpha_index_times_arr);
    Py_XDECREF(moment_coeffs_arr);
    Py_XDECREF(scaling_obj);
    Py_XDECREF(min_dist_obj);
    Py_XDECREF(max_dist_obj);
    Py_XDECREF(min_mag_obj);
    Py_XDECREF(max_mag_obj);
    Py_XDECREF(mag_basis_size_obj);
    Py_XDECREF(radial_basis_size_obj);
    Py_XDECREF(radial_coeffs_obj);
    Py_XDECREF(species_coeffs_obj);
    Py_XDECREF(species_count_obj);
    Py_XDECREF(radial_funcs_count_obj);
    Py_XDECREF(alpha_moments_count_obj);
    Py_XDECREF(alpha_moment_mapping_obj);
    Py_XDECREF(alpha_index_basic_obj);
    Py_XDECREF(alpha_index_basic_count_obj);
    Py_XDECREF(alpha_index_times_obj);
    Py_XDECREF(alpha_index_times_count_obj);
    Py_XDECREF(moment_coeffs_obj);
    return result;
}

/* ==========================================================================
 * Python wrapper for calc_mag_train_mgrad
 * ========================================================================== */
static PyObject *py_calc_mag_train_mgrad(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyArrayObject *js_arr, *rs_arr, *ms_arr, *itypes_arr, *jtypes_arr;
    PyObject *mtp_data_obj, *rbd_obj, *mbd_obj;

    static char *kwlist[] = {"js", "rs", "ms", "itypes", "jtypes", "mtp_data", "rbd", "mbd", NULL};

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "O!O!O!O!O!OOO",
            kwlist,
            &PyArray_Type, &js_arr,
            &PyArray_Type, &rs_arr,
            &PyArray_Type, &ms_arr,
            &PyArray_Type, &itypes_arr,
            &PyArray_Type, &jtypes_arr,
            &mtp_data_obj,
            &rbd_obj,
            &mbd_obj))
        return NULL;

    /* All owned references, NULL-initialized for safe Py_XDECREF in cleanup */
    PyObject *result = NULL;
    PyObject *scaling_obj = NULL, *min_dist_obj = NULL, *max_dist_obj = NULL;
    PyObject *min_mag_obj = NULL, *max_mag_obj = NULL;
    PyObject *mag_basis_size_obj = NULL, *radial_basis_size_obj = NULL;
    PyObject *radial_coeffs_obj = NULL, *species_coeffs_obj = NULL;
    PyObject *species_count_obj = NULL, *radial_funcs_count_obj = NULL;
    PyObject *alpha_moments_count_obj = NULL, *alpha_moment_mapping_obj = NULL;
    PyObject *alpha_index_basic_obj = NULL, *alpha_index_basic_count_obj = NULL;
    PyObject *alpha_index_times_obj = NULL, *alpha_index_times_count_obj = NULL;
    PyObject *moment_coeffs_obj = NULL;
    PyArrayObject *radial_coeffs_arr = NULL, *species_coeffs_arr = NULL;
    PyArrayObject *alpha_moment_mapping_arr = NULL;
    PyArrayObject *alpha_index_basic_arr = NULL, *alpha_index_times_arr = NULL;
    PyArrayObject *moment_coeffs_arr = NULL;
    PyArrayObject *energies_arr = NULL;

    PyObject *radial_basis_obj = PyObject_GetAttrString(mtp_data_obj, "radial_basis");
    if (radial_basis_obj)
    {
        min_dist_obj = PyObject_GetAttrString(radial_basis_obj, "min");
        max_dist_obj = PyObject_GetAttrString(radial_basis_obj, "max");
        radial_basis_size_obj = PyObject_GetAttrString(radial_basis_obj, "size");
        Py_DECREF(radial_basis_obj);
    }
    PyObject *magnetic_basis_obj = PyObject_GetAttrString(mtp_data_obj, "magnetic_basis");
    if (magnetic_basis_obj)
    {
        min_mag_obj = PyObject_GetAttrString(magnetic_basis_obj, "min");
        max_mag_obj = PyObject_GetAttrString(magnetic_basis_obj, "max");
        mag_basis_size_obj = PyObject_GetAttrString(magnetic_basis_obj, "size");
        Py_DECREF(magnetic_basis_obj);
    }

    scaling_obj = PyObject_GetAttrString(mtp_data_obj, "scaling");
    radial_coeffs_obj = PyObject_GetAttrString(mtp_data_obj, "radial_coeffs");
    species_coeffs_obj = PyObject_GetAttrString(mtp_data_obj, "species_coeffs");
    species_count_obj = PyObject_GetAttrString(mtp_data_obj, "species_count");
    radial_funcs_count_obj = PyObject_GetAttrString(mtp_data_obj, "radial_funcs_count");
    alpha_moments_count_obj = PyObject_GetAttrString(mtp_data_obj, "alpha_moments_count");
    alpha_moment_mapping_obj = PyObject_GetAttrString(mtp_data_obj, "alpha_moment_mapping");
    alpha_index_basic_obj = PyObject_GetAttrString(mtp_data_obj, "alpha_index_basic");
    alpha_index_basic_count_obj = PyObject_GetAttrString(mtp_data_obj, "alpha_index_basic_count");
    alpha_index_times_obj = PyObject_GetAttrString(mtp_data_obj, "alpha_index_times");
    alpha_index_times_count_obj = PyObject_GetAttrString(mtp_data_obj, "alpha_index_times_count");
    moment_coeffs_obj = PyObject_GetAttrString(mtp_data_obj, "moment_coeffs");

    if (!scaling_obj || !min_dist_obj || !max_dist_obj || !min_mag_obj || !max_mag_obj ||
        !mag_basis_size_obj || !radial_basis_size_obj || !radial_coeffs_obj || !species_coeffs_obj || !species_count_obj ||
        !radial_funcs_count_obj || !alpha_moments_count_obj || !alpha_moment_mapping_obj ||
        !alpha_index_basic_obj || !alpha_index_basic_count_obj || !alpha_index_times_obj ||
        !alpha_index_times_count_obj || !moment_coeffs_obj)
        goto cleanup;

    double scaling = PyFloat_AsDouble(scaling_obj);
    double min_dist = PyFloat_AsDouble(min_dist_obj);
    double max_dist = PyFloat_AsDouble(max_dist_obj);
    double min_mag = PyFloat_AsDouble(min_mag_obj);
    double max_mag = PyFloat_AsDouble(max_mag_obj);
    int mag_basis_size = (int)PyLong_AsLong(mag_basis_size_obj);
    int species_count = (int)PyLong_AsLong(species_count_obj);
    int radial_funcs_count = (int)PyLong_AsLong(radial_funcs_count_obj);
    int alpha_moments_count = (int)PyLong_AsLong(alpha_moments_count_obj);
    int n_basic = (int)PyLong_AsLong(alpha_index_basic_count_obj);
    int n_times = (int)PyLong_AsLong(alpha_index_times_count_obj);

    radial_coeffs_arr = (PyArrayObject *)PyArray_FROM_OTF(radial_coeffs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    species_coeffs_arr = (PyArrayObject *)PyArray_FROM_OTF(species_coeffs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    alpha_moment_mapping_arr = (PyArrayObject *)PyArray_FROM_OTF(alpha_moment_mapping_obj, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);
    alpha_index_basic_arr = (PyArrayObject *)PyArray_FROM_OTF(alpha_index_basic_obj, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);
    alpha_index_times_arr = (PyArrayObject *)PyArray_FROM_OTF(alpha_index_times_obj, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);
    moment_coeffs_arr = (PyArrayObject *)PyArray_FROM_OTF(moment_coeffs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!radial_coeffs_arr || !species_coeffs_arr || !alpha_moment_mapping_arr ||
        !alpha_index_basic_arr || !alpha_index_times_arr || !moment_coeffs_arr)
        goto cleanup;

    if (!require_int32(js_arr, "js") ||
        !require_int32(itypes_arr, "itypes") ||
        !require_int32(jtypes_arr, "jtypes") ||
        !require_int32(alpha_moment_mapping_arr, "alpha_moment_mapping") ||
        !require_int32(alpha_index_basic_arr, "alpha_index_basic") ||
        !require_int32(alpha_index_times_arr, "alpha_index_times"))
        goto cleanup;

    int n_atoms = (int)PyArray_DIM(rs_arr, 0);
    int n_neighbors = (int)PyArray_DIM(rs_arr, 1);
    int radial_basis_size = (int)PyLong_AsLong(radial_basis_size_obj);
    int n_alpha_scalar = (int)PyArray_DIM(alpha_moment_mapping_arr, 0);

    /* rbd/mbd borrowed views — obj decref'd inline, arr not owned */
    PyObject *rbd_values_obj = PyObject_GetAttrString(rbd_obj, "values");
    PyArrayObject *rbd_values_arr = rbd_values_obj ? get_inplace_double(rbd_values_obj, "rbd.values") : NULL;
    Py_XDECREF(rbd_values_obj);
    PyObject *rbd_dqdris_obj = PyObject_GetAttrString(rbd_obj, "dqdris");
    PyArrayObject *rbd_dqdris_arr = rbd_dqdris_obj ? get_inplace_double(rbd_dqdris_obj, "rbd.dqdris") : NULL;
    Py_XDECREF(rbd_dqdris_obj);
    PyObject *rbd_dqdmis_obj = PyObject_GetAttrString(rbd_obj, "dqdmis");
    PyArrayObject *rbd_dqdmis_arr = rbd_dqdmis_obj ? get_inplace_double(rbd_dqdmis_obj, "rbd.dqdmis") : NULL;
    Py_XDECREF(rbd_dqdmis_obj);
    PyObject *rbd_dqdeps_obj = PyObject_GetAttrString(rbd_obj, "dqdeps");
    PyArrayObject *rbd_dqdeps_arr = rbd_dqdeps_obj ? get_inplace_double(rbd_dqdeps_obj, "rbd.dqdeps") : NULL;
    Py_XDECREF(rbd_dqdeps_obj);
    PyObject *mbd_vatoms_obj = PyObject_GetAttrString(mbd_obj, "vatoms");
    PyArrayObject *mbd_vatoms_arr = mbd_vatoms_obj ? get_inplace_double(mbd_vatoms_obj, "mbd.vatoms") : NULL;
    Py_XDECREF(mbd_vatoms_obj);
    PyObject *mbd_dbdris_obj = PyObject_GetAttrString(mbd_obj, "dbdris");
    PyArrayObject *mbd_dbdris_arr = mbd_dbdris_obj ? get_inplace_double(mbd_dbdris_obj, "mbd.dbdris") : NULL;
    Py_XDECREF(mbd_dbdris_obj);
    PyObject *mbd_dbdmis_obj = PyObject_GetAttrString(mbd_obj, "dbdmis");
    PyArrayObject *mbd_dbdmis_arr = mbd_dbdmis_obj ? get_inplace_double(mbd_dbdmis_obj, "mbd.dbdmis") : NULL;
    Py_XDECREF(mbd_dbdmis_obj);
    PyObject *mbd_dbdeps_obj = PyObject_GetAttrString(mbd_obj, "dbdeps");
    PyArrayObject *mbd_dbdeps_arr = mbd_dbdeps_obj ? get_inplace_double(mbd_dbdeps_obj, "mbd.dbdeps") : NULL;
    Py_XDECREF(mbd_dbdeps_obj);
    PyObject *mbd_dvdcs_obj = PyObject_GetAttrString(mbd_obj, "dvdcs");
    PyArrayObject *mbd_dvdcs_arr = mbd_dvdcs_obj ? get_inplace_double(mbd_dvdcs_obj, "mbd.dvdcs") : NULL;
    Py_XDECREF(mbd_dvdcs_obj);
    PyObject *mbd_dgdcs_obj = PyObject_GetAttrString(mbd_obj, "dgdcs");
    PyArrayObject *mbd_dgdcs_arr = mbd_dgdcs_obj ? get_inplace_double(mbd_dgdcs_obj, "mbd.dgdcs") : NULL;
    Py_XDECREF(mbd_dgdcs_obj);
    PyObject *mbd_dgmdcs_obj = PyObject_GetAttrString(mbd_obj, "dgmdcs");
    PyArrayObject *mbd_dgmdcs_arr = mbd_dgmdcs_obj ? get_inplace_double(mbd_dgmdcs_obj, "mbd.dgmdcs") : NULL;
    Py_XDECREF(mbd_dgmdcs_obj);
    PyObject *mbd_dsdcs_obj = PyObject_GetAttrString(mbd_obj, "dsdcs");
    PyArrayObject *mbd_dsdcs_arr = mbd_dsdcs_obj ? get_inplace_double(mbd_dsdcs_obj, "mbd.dsdcs") : NULL;
    Py_XDECREF(mbd_dsdcs_obj);

    if (!rbd_values_arr || !rbd_dqdris_arr || !rbd_dqdmis_arr || !rbd_dqdeps_arr ||
        !mbd_vatoms_arr || !mbd_dbdris_arr || !mbd_dbdmis_arr || !mbd_dbdeps_arr ||
        !mbd_dvdcs_arr || !mbd_dgdcs_arr || !mbd_dgmdcs_arr || !mbd_dsdcs_arr)
        goto cleanup;

    npy_intp dims_energy[1] = {n_atoms};
    energies_arr = (PyArrayObject *)PyArray_ZEROS(1, dims_energy, NPY_DOUBLE, 0);

    if (!energies_arr)
    {
        if (!PyErr_Occurred())
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate output arrays");
        goto cleanup;
    }

    calc_mag_train_mgrad(
        n_atoms, n_neighbors,
        (double *)PyArray_DATA(rs_arr),
        (int *)PyArray_DATA(js_arr),
        (double *)PyArray_DATA(ms_arr),
        (int *)PyArray_DATA(itypes_arr),
        (int *)PyArray_DATA(jtypes_arr),
        scaling, min_dist, max_dist,
        min_mag, max_mag,
        radial_basis_size,
        mag_basis_size,
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
        (double *)PyArray_DATA(rbd_values_arr),
        (double *)PyArray_DATA(rbd_dqdris_arr),
        (double *)PyArray_DATA(rbd_dqdmis_arr),
        (double *)PyArray_DATA(rbd_dqdeps_arr),
        (double *)PyArray_DATA(mbd_vatoms_arr),
        (double *)PyArray_DATA(mbd_dbdris_arr),
        (double *)PyArray_DATA(mbd_dbdmis_arr),
        (double *)PyArray_DATA(mbd_dbdeps_arr),
        (double *)PyArray_DATA(mbd_dvdcs_arr),
        (double *)PyArray_DATA(mbd_dgdcs_arr),
        (double *)PyArray_DATA(mbd_dgmdcs_arr),
        (double *)PyArray_DATA(mbd_dsdcs_arr));

    result = (PyObject *)energies_arr;
    energies_arr = NULL;

cleanup:
    Py_XDECREF(energies_arr);
    Py_XDECREF(radial_coeffs_arr);
    Py_XDECREF(species_coeffs_arr);
    Py_XDECREF(alpha_moment_mapping_arr);
    Py_XDECREF(alpha_index_basic_arr);
    Py_XDECREF(alpha_index_times_arr);
    Py_XDECREF(moment_coeffs_arr);
    Py_XDECREF(scaling_obj);
    Py_XDECREF(min_dist_obj);
    Py_XDECREF(max_dist_obj);
    Py_XDECREF(min_mag_obj);
    Py_XDECREF(max_mag_obj);
    Py_XDECREF(mag_basis_size_obj);
    Py_XDECREF(radial_basis_size_obj);
    Py_XDECREF(radial_coeffs_obj);
    Py_XDECREF(species_coeffs_obj);
    Py_XDECREF(species_count_obj);
    Py_XDECREF(radial_funcs_count_obj);
    Py_XDECREF(alpha_moments_count_obj);
    Py_XDECREF(alpha_moment_mapping_obj);
    Py_XDECREF(alpha_index_basic_obj);
    Py_XDECREF(alpha_index_basic_count_obj);
    Py_XDECREF(alpha_index_times_obj);
    Py_XDECREF(alpha_index_times_count_obj);
    Py_XDECREF(moment_coeffs_obj);
    return result;
}

static PyMethodDef module_methods[] = {
    {"calc_mag_run", (PyCFunction)py_calc_mag_run, METH_VARARGS | METH_KEYWORDS,
     "Calculate energies, gradients, and magnetic gradients for run mode"},
    {"calc_mag_train", (PyCFunction)py_calc_mag_train, METH_VARARGS | METH_KEYWORDS,
     "Calculate energies and populate RBD/MBD for training mode"},
    {"calc_mag_train_mgrad", (PyCFunction)py_calc_mag_train_mgrad, METH_VARARGS | METH_KEYWORDS,
     "Calculate energies and populate RBD/MBD for train_mgrad mode"},
    {"calc_forces_from_gradient", py_calc_forces_from_gradient, METH_VARARGS,
     "Compute forces from gradients"},
    {"calc_mgrad_from_gradient", py_calc_mgrad_from_gradient, METH_VARARGS,
     "Compute magnetic gradients from per-pair gradients"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_mmtp_cext",
    NULL,
    -1,
    module_methods,
};

PyMODINIT_FUNC PyInit__mmtp_cext(void)
{
    import_array();
    return PyModule_Create(&moduledef);
}
