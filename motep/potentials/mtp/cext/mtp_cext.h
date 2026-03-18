#ifndef MTP_CEXT_H
#define MTP_CEXT_H

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <string.h>

/* ============================================================================
 * C struct definitions for basis data (matching Python dataclasses)
 * ============================================================================ */

/* Radial basis data structure */
typedef struct
{
    double *values; /* (species_count, species_count, radial_basis_size) */
    double *dqdris; /* (species_count, species_count, radial_basis_size, n_atoms, 3) */
    double *dqdeps; /* (species_count, species_count, radial_basis_size, 3, 3) */
} RadialBasisData;

/* Moment basis data structure */
typedef struct
{
    double *vatoms; /* (alpha_scalar_moments, n_atoms) */
    double *dbdris; /* (alpha_scalar_moments, n_atoms, 3) */
    double *dbdeps; /* (alpha_scalar_moments, 3, 3) */
    double *dedcs;  /* (species_count, species_count, radial_funcs_count, radial_basis_size) */
    double *dgdcs;  /* (species_count, species_count, radial_funcs_count, radial_basis_size, n_atoms, 3) */
    double *dsdcs;  /* (species_count, species_count, radial_funcs_count, radial_basis_size, 3, 3) */
} MomentBasisData;

/* ============================================================================
 * Simplified C extension for MTP run mode calculations
 *
 * Focus: Core calculation loop only (no training mode yet)
 * Input: Atomic coordinates, types, and MTP parameters
 * Output: Per-atom energies and gradients w.r.t. atomic coordinates
 * ============================================================================ */

/* Main calculation for run mode
 * Parameters extracted from mtp_data and passed individually for clarity
 */
void calc_run(
    int n_atoms, int n_neighbors,
    const double *rij, /* (n_atoms, n_neighbors, 3) */
    const int *itypes, /* (n_atoms) */
    const int *jtype,  /* (n_atoms, n_neighbors) */
    /* Radial basis parameters */
    double scaling, double min_dist, double max_dist,
    int radial_basis_size,
    const double *radial_coeffs, /* (species_count, species_count, radial_funcs_count, radial_basis_size) */
    int species_count, int radial_funcs_count,
    /* Species baseline */
    const double *species_coeffs, /* (species_count) */
    /* Moment tensor information */
    int alpha_moments_count,
    const int *alpha_moment_mapping, /* (alpha_scalar_moments) */
    int n_alpha_scalar,
    const int *alpha_index_basic, /* (n_basic, 4): [mu, xpow, ypow, zpow] */
    int n_basic,
    const int *alpha_index_times, /* (n_times, 4): [i1, i2, mult, i3] */
    int n_times,
    const double *moment_coeffs, /* (alpha_scalar_moments) */
    /* Output */
    double *energies,    /* (n_atoms) */
    double *gradient,    /* (n_atoms, n_neighbors, 3) */
    double *mbd_vatoms); /* (n_alpha_scalar, n_atoms) */

/* Training mode calculation - computes energies and parameter derivatives
 * Takes preallocated RadialBasisData and MomentBasisData structures.
 * Writes results directly into provided buffer pointers.
 */
void calc_train(
    int n_atoms, int n_neighbors,
    const double *rij,
    const int *js, /* (n_atoms, n_neighbors) - neighbor indices for mapping */
    const int *itypes,
    const int *jtype,
    double scaling, double min_dist, double max_dist,
    int radial_basis_size,
    const double *radial_coeffs,
    int species_count, int radial_funcs_count,
    const double *species_coeffs,
    int alpha_moments_count,
    const int *alpha_moment_mapping,
    int n_alpha_scalar,
    const int *alpha_index_basic,
    int n_basic,
    const int *alpha_index_times,
    int n_times,
    const double *moment_coeffs,
    /* Output: per-atom energies */
    double *energies,
    /* Preallocated data structures for basis values and derivatives */
    RadialBasisData *rbd,
    MomentBasisData *mbd);

/* Calculate forces from gradients */
void calc_forces_from_gradient(
    const double *gradient, /* (n_atoms, n_neighbors, 3) */
    const int *js,          /* (n_atoms, n_neighbors) */
    int n_atoms, int n_neighbors,
    double *forces); /* (n_atoms, 3) */

#endif /* MTP_CEXT_H */
