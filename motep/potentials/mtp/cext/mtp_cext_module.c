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
        return NULL;

    /* All owned references, NULL-initialized for safe Py_XDECREF in cleanup */
    PyObject *result = NULL;
    PyObject *scaling_obj = NULL, *min_dist_obj = NULL, *max_dist_obj = NULL;
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

    PyObject *radial_basis_obj = PyObject_GetAttrString(mtp_data_obj, "radial_basis");
    if (radial_basis_obj)
    {
        min_dist_obj = PyObject_GetAttrString(radial_basis_obj, "min");
        max_dist_obj = PyObject_GetAttrString(radial_basis_obj, "max");
        Py_DECREF(radial_basis_obj);
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

    if (!scaling_obj || !min_dist_obj || !max_dist_obj || !radial_coeffs_obj ||
        !species_coeffs_obj || !species_count_obj || !radial_funcs_count_obj ||
        !alpha_moments_count_obj || !alpha_moment_mapping_obj || !alpha_index_basic_obj ||
        !alpha_index_basic_count_obj || !alpha_index_times_obj || !alpha_index_times_count_obj ||
        !moment_coeffs_obj)
        goto cleanup;

    double scaling = PyFloat_AsDouble(scaling_obj);
    double min_dist = PyFloat_AsDouble(min_dist_obj);
    double max_dist = PyFloat_AsDouble(max_dist_obj);
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

    if (!require_int32(itypes_arr, "itypes") ||
        !require_int32(jtypes_arr, "jtypes") ||
        !require_int32(alpha_moment_mapping_arr, "alpha_moment_mapping") ||
        !require_int32(alpha_index_basic_arr, "alpha_index_basic") ||
        !require_int32(alpha_index_times_arr, "alpha_index_times"))
        goto cleanup;

    int n_atoms = (int)PyArray_DIM(rs_arr, 0);
    int n_neighbors = (int)PyArray_DIM(rs_arr, 1);
    int radial_basis_size = (int)PyArray_DIM(radial_coeffs_arr, 3);
    int n_alpha_scalar = (int)PyArray_DIM(alpha_moment_mapping_arr, 0);

    /* mbd_vatoms_arr is a borrowed view — obj decref'd inline, arr not owned */
    PyObject *mbd_vatoms_obj = PyObject_GetAttrString(mbd_obj, "vatoms");
    PyArrayObject *mbd_vatoms_arr = mbd_vatoms_obj ? get_inplace_double(mbd_vatoms_obj, "mbd.vatoms") : NULL;
    Py_XDECREF(mbd_vatoms_obj);
    if (!mbd_vatoms_arr)
        goto cleanup;

    npy_intp dims_energy[1] = {n_atoms};
    npy_intp dims_grad[3] = {n_atoms, n_neighbors, 3};
    energies_arr = (PyArrayObject *)PyArray_ZEROS(1, dims_energy, NPY_DOUBLE, 0);
    gradient_arr = (PyArrayObject *)PyArray_ZEROS(3, dims_grad, NPY_DOUBLE, 0);

    if (!energies_arr || !gradient_arr)
    {
        if (!PyErr_Occurred())
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate output arrays");
        goto cleanup;
    }

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

    /* N steals refs — NULL out so cleanup's Py_XDECREF is a no-op for these */
    result = Py_BuildValue("(NN)", energies_arr, gradient_arr);
    energies_arr = gradient_arr = NULL;

cleanup:
    Py_XDECREF(energies_arr);
    Py_XDECREF(gradient_arr);
    Py_XDECREF(radial_coeffs_arr);
    Py_XDECREF(species_coeffs_arr);
    Py_XDECREF(alpha_moment_mapping_arr);
    Py_XDECREF(alpha_index_basic_arr);
    Py_XDECREF(alpha_index_times_arr);
    Py_XDECREF(moment_coeffs_arr);
    Py_XDECREF(scaling_obj);
    Py_XDECREF(min_dist_obj);
    Py_XDECREF(max_dist_obj);
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
        return NULL;

    /* All owned references, NULL-initialized for safe Py_XDECREF in cleanup */
    PyObject *result = NULL;
    PyObject *scaling_obj = NULL, *min_dist_obj = NULL, *max_dist_obj = NULL;
    PyObject *radial_coeffs_obj = NULL, *species_coeffs_obj = NULL;
    PyObject *species_count_obj = NULL, *radial_funcs_count_obj = NULL;
    PyObject *alpha_moments_count_obj = NULL, *alpha_moment_mapping_obj = NULL;
    PyObject *alpha_index_basic_obj = NULL, *alpha_index_basic_count_obj = NULL;
    PyObject *alpha_index_times_obj = NULL, *alpha_index_times_count_obj = NULL;
    PyObject *moment_coeffs_obj = NULL, *alpha_scalar_moments_obj = NULL;
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
        Py_DECREF(radial_basis_obj);
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
    alpha_scalar_moments_obj = PyObject_GetAttrString(mtp_data_obj, "alpha_scalar_moments");

    if (!scaling_obj || !min_dist_obj || !max_dist_obj || !radial_coeffs_obj ||
        !species_coeffs_obj || !species_count_obj || !radial_funcs_count_obj ||
        !alpha_moments_count_obj || !alpha_moment_mapping_obj || !alpha_index_basic_obj ||
        !alpha_index_basic_count_obj || !alpha_index_times_obj || !alpha_index_times_count_obj ||
        !moment_coeffs_obj || !alpha_scalar_moments_obj)
        goto cleanup;

    double scaling = PyFloat_AsDouble(scaling_obj);
    double min_dist = PyFloat_AsDouble(min_dist_obj);
    double max_dist = PyFloat_AsDouble(max_dist_obj);
    int species_count = (int)PyLong_AsLong(species_count_obj);
    int radial_funcs_count = (int)PyLong_AsLong(radial_funcs_count_obj);
    int alpha_moments_count = (int)PyLong_AsLong(alpha_moments_count_obj);
    int n_basic = (int)PyLong_AsLong(alpha_index_basic_count_obj);
    int n_times = (int)PyLong_AsLong(alpha_index_times_count_obj);
    int n_alpha_scalar = (int)PyLong_AsLong(alpha_scalar_moments_obj);

    radial_coeffs_arr = (PyArrayObject *)PyArray_FROM_OTF(radial_coeffs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    species_coeffs_arr = (PyArrayObject *)PyArray_FROM_OTF(species_coeffs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    alpha_moment_mapping_arr = (PyArrayObject *)PyArray_FROM_OTF(alpha_moment_mapping_obj, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);
    alpha_index_basic_arr = (PyArrayObject *)PyArray_FROM_OTF(alpha_index_basic_obj, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);
    alpha_index_times_arr = (PyArrayObject *)PyArray_FROM_OTF(alpha_index_times_obj, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY);
    moment_coeffs_arr = (PyArrayObject *)PyArray_FROM_OTF(moment_coeffs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!radial_coeffs_arr || !species_coeffs_arr || !alpha_moment_mapping_arr ||
        !alpha_index_basic_arr || !alpha_index_times_arr || !moment_coeffs_arr)
        goto cleanup;

    if (!require_int32(itypes_arr, "itypes") ||
        !require_int32(jtypes_arr, "jtypes") ||
        !require_int32(alpha_moment_mapping_arr, "alpha_moment_mapping") ||
        !require_int32(alpha_index_basic_arr, "alpha_index_basic") ||
        !require_int32(alpha_index_times_arr, "alpha_index_times"))
        goto cleanup;

    int n_atoms = (int)PyArray_DIM(rs_arr, 0);
    int n_neighbors = (int)PyArray_DIM(rs_arr, 1);
    int radial_basis_size = (int)PyArray_DIM(radial_coeffs_arr, 3);

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

    npy_intp dims_energies[1] = {n_atoms};
    energies_arr = (PyArrayObject *)PyArray_ZEROS(1, dims_energies, NPY_DOUBLE, 0);

    if (!energies_arr)
    {
        if (!PyErr_Occurred())
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate output arrays");
        goto cleanup;
    }

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
    Py_XDECREF(alpha_scalar_moments_obj);
    return result;
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
