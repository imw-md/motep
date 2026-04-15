#ifndef MMTP_CEXT_H
#define MMTP_CEXT_H

#include <math.h>
#include <string.h>
#include <stdlib.h>

/* Main calculation for magnetic run mode */
void calc_mag_run(
    int n_atoms, int n_neighbors,
    const double *rs,  /* (n_atoms, n_neighbors, 3) */
    const int *js,     /* (n_atoms, n_neighbors) */
    const double *ms,  /* (n_atoms) */
    const int *itypes, /* (n_atoms) */
    const int *jtype,  /* (n_atoms, n_neighbors) */
    double scaling, double min_dist, double max_dist,
    double min_mag, double max_mag,
    int radial_basis_size,
    int mag_basis_size,
    const double *radial_coeffs, /* (species_count, species_count, radial_funcs_count, radial_basis_size * mag_basis_size^2) */
    int species_count, int radial_funcs_count,
    const double *species_coeffs, /* (species_count) */
    int alpha_moments_count,
    const int *alpha_moment_mapping, /* (alpha_scalar_moments) */
    int n_alpha_scalar,
    const int *alpha_index_basic, /* (n_basic, 4): [mu, xpow, ypow, zpow] */
    int n_basic,
    const int *alpha_index_times, /* (n_times, 4): [i1, i2, mult, i3] */
    int n_times,
    const double *moment_coeffs, /* (alpha_scalar_moments) */
    double *energies,            /* (n_atoms) */
    double *gradient,            /* (n_atoms, n_neighbors, 3) */
    double *grad_mag_i,          /* (n_atoms, n_neighbors) */
    double *grad_mag_j,          /* (n_atoms, n_neighbors) */
    double *mbd_vatoms);         /* (n_alpha_scalar, n_atoms) */

/* Calculate forces from gradients */
void calc_forces_from_gradient(
    const double *gradient, /* (n_atoms, n_neighbors, 3) */
    const int *js,          /* (n_atoms, n_neighbors) */
    int n_atoms, int n_neighbors,
    double *forces); /* (n_atoms, 3) */

/* Calculate magnetic gradients from per-pair gradients */
void calc_mgrad_from_gradient(
    const double *grad_mag_i, /* (n_atoms, n_neighbors) */
    const double *grad_mag_j, /* (n_atoms, n_neighbors) */
    const int *js,            /* (n_atoms, n_neighbors) */
    int n_atoms, int n_neighbors,
    double *mgrad); /* (n_atoms) */

/* Training mode calculation */
void calc_mag_train(
    int n_atoms, int n_neighbors,
    const double *rs,
    const int *js,
    const double *ms,
    const int *itypes,
    const int *jtype,
    double scaling, double min_dist, double max_dist,
    double min_mag, double max_mag,
    int radial_basis_size,
    int mag_basis_size,
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
    double *energies,
    double *rbd_values,
    double *rbd_dqdris,
    double *rbd_dqdeps,
    double *mbd_vatoms,
    double *mbd_dbdris,
    double *mbd_dbdeps,
    double *mbd_dedcs,
    double *mbd_dgdcs,
    double *mbd_dsdcs);

/* Training mode calculation with magnetic gradients */
void calc_mag_train_mgrad(
    int n_atoms, int n_neighbors,
    const double *rs,
    const int *js,
    const double *ms,
    const int *itypes,
    const int *jtype,
    double scaling, double min_dist, double max_dist,
    double min_mag, double max_mag,
    int radial_basis_size,
    int mag_basis_size,
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
    double *energies,
    double *rbd_values,
    double *rbd_dqdris,
    double *rbd_dqdmis,
    double *rbd_dqdeps,
    double *mbd_vatoms,
    double *mbd_dbdris,
    double *mbd_dbdmis,
    double *mbd_dbdeps,
    double *mbd_dedcs,
    double *mbd_dgdcs,
    double *mbd_dgmdcs,
    double *mbd_dsdcs);

#endif /* MMTP_CEXT_H */
