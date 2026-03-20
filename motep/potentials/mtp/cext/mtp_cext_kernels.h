#ifndef MTP_CEXT_KERNELS_H
#define MTP_CEXT_KERNELS_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ==========================================================================
 * Helper: Calculate Chebyshev polynomials and derivatives
 * ========================================================================== */
static inline void chebyshev_basis(
    double r_abs, int n_basis,
    double scaling, double min_dist, double max_dist,
    double *basis_vals, double *basis_derivs)
{
    /* Scale distance to [-1, 1] range */
    double x = 2.0 * (r_abs - min_dist) / (max_dist - min_dist) - 1.0;

    double scale_factor = 2.0 / (max_dist - min_dist);

    /* Apply smooth cutoff */
    double diff = max_dist - r_abs;
    double smooth_val = (diff > 0.0) ? scaling * diff * diff : 0.0;
    double smooth_deriv = (diff > 0.0) ? -2.0 * scaling * diff : 0.0;
    // double smooth_val = scaling * diff * diff;
    // double smooth_deriv = -2.0 * scaling * diff;

    /* T_0 = 1, T_0' = 0 */
    basis_vals[0] = smooth_val;
    basis_derivs[0] = smooth_deriv;

    /* T_1 = x, T_1' = 1 */
    basis_vals[1] = x * smooth_val;
    basis_derivs[1] = scale_factor * smooth_val + x * smooth_deriv;

    /* Recurrence: T_n = 2*x*T_{n-1} - T_{n-2} */
    for (int i = 2; i < n_basis; i++)
    {
        double t_curr = 2.0 * x * basis_vals[i - 1] - basis_vals[i - 2];
        double dt_curr = 2.0 * basis_vals[i - 1] * scale_factor +
                         2.0 * x * basis_derivs[i - 1] - basis_derivs[i - 2];

        basis_vals[i] = t_curr;
        basis_derivs[i] = dt_curr;
    }
}

/* ==========================================================================
 * Helper: Sum radial terms for given atom pair types
 * Processes basis input (either values or derivatives) into output
 * radial_coeffs[itype, jtype, irf, irb] layout in C-order (row-major)
 * input_basis layout: shape (n_basis, n_neighbors) in row-major order
 * ========================================================================== */
static inline void sum_radial_basis(
    int itype, const int *jtypes, int n_neighbors,
    const double *input_basis, int n_basis,
    const double *radial_coeffs, int species_count, int radial_funcs_count,
    double *output)
{
    int rfc = radial_funcs_count;
    int rbs = n_basis;

    /* Initialize output array */
    memset(output, 0, radial_funcs_count * n_neighbors * sizeof(double));

    for (int j = 0; j < n_neighbors; j++)
    {
        int jtype = jtypes[j];

        for (int irf = 0; irf < rfc; irf++)
        {
            double sum = 0.0;

            for (int irb = 0; irb < rbs; irb++)
            {
                /* Access: radial_coeffs[itype, jtype, irf, irb] */
                int coeff_idx = ((itype * species_count + jtype) * rfc + irf) * rbs + irb;
                /* Access input_basis[irb, j] via flat index: irb * n_neighbors + j */
                sum += radial_coeffs[coeff_idx] * input_basis[irb * n_neighbors + j];
            }

            output[irf * n_neighbors + j] = sum;
        }
    }
}

/* ==========================================================================
 * Calculate basic moment values and their jacobians w.r.t. atomic positions
 * ========================================================================== */
static inline void calc_basic_moments(
    int n_neighbors,
    const double *r_abs,
    const double *r_unit,
    const double *rf_vals,
    const double *drf_drs,
    const int *alpha_index_basic,
    int n_basic,
    double *moments_basic,
    double *moment_jacobian)
{

    for (int i_aib = 0; i_aib < n_basic; i_aib++)
    {
        int mu = alpha_index_basic[i_aib * 4 + 0];
        int xpow = alpha_index_basic[i_aib * 4 + 1];
        int ypow = alpha_index_basic[i_aib * 4 + 2];
        int zpow = alpha_index_basic[i_aib * 4 + 3];
        int ang_pow = xpow + ypow + zpow;

        for (int k = 0; k < n_neighbors; k++)
        {
            /* Get radial function value and derivative */
            double rf = rf_vals[mu * n_neighbors + k];
            double drf_dr = drf_drs[mu * n_neighbors + k];

            /* Calculate angular terms (powers of unit vector components) */
            double rx_pow = 1.0, ry_pow = 1.0, rz_pow = 1.0;
            for (int p = 0; p < xpow; p++)
                rx_pow *= r_unit[k * 3 + 0];
            for (int p = 0; p < ypow; p++)
                ry_pow *= r_unit[k * 3 + 1];
            for (int p = 0; p < zpow; p++)
                rz_pow *= r_unit[k * 3 + 2];

            double angular = rx_pow * ry_pow * rz_pow;
            double moment_val = rf * angular;

            /* Add to moment sum */
            moments_basic[i_aib] += moment_val;

            /* Calculate jacobian: d(moment)/d(r_ij) */
            if (r_abs[k] > 1e-10)
            {
                /* Radial contribution: d(rf*angular)/dr_ij */
                double radial_jac = angular * (drf_dr - ang_pow * rf / r_abs[k]);
                moment_jacobian[(i_aib * n_neighbors + k) * 3 + 0] = r_unit[k * 3 + 0] * radial_jac;
                moment_jacobian[(i_aib * n_neighbors + k) * 3 + 1] = r_unit[k * 3 + 1] * radial_jac;
                moment_jacobian[(i_aib * n_neighbors + k) * 3 + 2] = r_unit[k * 3 + 2] * radial_jac;

                /* Angular contribution from x power: d(rf * rx^xpow * ry^ypow * rz^zpow)/d(r_ij) */
                if (xpow > 0)
                {
                    double rx_prev = 1.0;
                    for (int p = 0; p < xpow - 1; p++)
                        rx_prev *= r_unit[k * 3 + 0];
                    double ang_contrib = rf * xpow * rx_prev * ry_pow * rz_pow / r_abs[k];
                    moment_jacobian[(i_aib * n_neighbors + k) * 3 + 0] += ang_contrib;
                }

                /* Angular contribution from y power */
                if (ypow > 0)
                {
                    double ry_prev = 1.0;
                    for (int p = 0; p < ypow - 1; p++)
                        ry_prev *= r_unit[k * 3 + 1];
                    double ang_contrib = rf * rx_pow * ypow * ry_prev * rz_pow / r_abs[k];
                    moment_jacobian[(i_aib * n_neighbors + k) * 3 + 1] += ang_contrib;
                }

                /* Angular contribution from z power */
                if (zpow > 0)
                {
                    double rz_prev = 1.0;
                    for (int p = 0; p < zpow - 1; p++)
                        rz_prev *= r_unit[k * 3 + 2];
                    double ang_contrib = rf * rx_pow * ry_pow * zpow * rz_prev / r_abs[k];
                    moment_jacobian[(i_aib * n_neighbors + k) * 3 + 2] += ang_contrib;
                }
            }
        }
    }
}

/* ==========================================================================
 * Compute jacobians of basic moments w.r.t. radial coefficients
 * moment_jac_cs[i_aib, jtype, mu, i_rb] = rb_values[i_rb, j] * angular_factor
 * where mu is the radial function index from alpha_index_basic
 * Following numba's _calc_moment_basic_with_jacobian_radial_coeffs pattern
 * ========================================================================== */
static inline void calc_basic_moments_jac_radial_coeffs(
    int n_neighbors,
    const double *r_abs,
    const double *r_unit,
    const double *rb_vals,
    const double *rb_derivs,
    const int *alpha_index_basic,
    int n_basic,
    int species_count,
    const int *jtypes,
    int radial_funcs_count,
    int radial_basis_size,
    double *moment_jac_cs,
    double *moment_jac_rc)
{
    int rbs = radial_basis_size;
    int rfc = radial_funcs_count;

    for (int i_aib = 0; i_aib < n_basic; i_aib++)
    {
        int mu = alpha_index_basic[i_aib * 4 + 0];
        int xpow = alpha_index_basic[i_aib * 4 + 1];
        int ypow = alpha_index_basic[i_aib * 4 + 2];
        int zpow = alpha_index_basic[i_aib * 4 + 3];
        int xyzpow = xpow + ypow + zpow;

        for (int k = 0; k < n_neighbors; k++)
        {
            int jtype = jtypes[k];

            /* Calculate angular terms */
            double rx_pow = 1.0, ry_pow = 1.0, rz_pow = 1.0;
            for (int p = 0; p < xpow; p++)
                rx_pow *= r_unit[k * 3 + 0];
            for (int p = 0; p < ypow; p++)
                ry_pow *= r_unit[k * 3 + 1];
            for (int p = 0; p < zpow; p++)
                rz_pow *= r_unit[k * 3 + 2];

            double angular = rx_pow * ry_pow * rz_pow;

            /* For each radial basis index */
            for (int i_rb = 0; i_rb < rbs; i_rb++)
            {
                double rb_val = rb_vals[i_rb * n_neighbors + k];

                int idx_cs = ((i_aib * species_count + jtype) * rfc + mu) * rbs + i_rb;
                moment_jac_cs[idx_cs] += rb_val * angular;

                if (r_abs[k] > 1e-10)
                {
                    double rb_der = rb_derivs[i_rb * n_neighbors + k];

                    double der0 = r_unit[k * 3 + 0] * angular *
                                  (rb_der - xyzpow * rb_val / r_abs[k]);
                    double der1 = r_unit[k * 3 + 1] * angular *
                                  (rb_der - xyzpow * rb_val / r_abs[k]);
                    double der2 = r_unit[k * 3 + 2] * angular *
                                  (rb_der - xyzpow * rb_val / r_abs[k]);

                    if (xpow > 0)
                    {
                        double rx_prev = 1.0;
                        for (int p = 0; p < xpow - 1; p++)
                            rx_prev *= r_unit[k * 3 + 0];
                        der0 += rb_val * xpow * rx_prev * ry_pow * rz_pow / r_abs[k];
                    }
                    if (ypow > 0)
                    {
                        double ry_prev = 1.0;
                        for (int p = 0; p < ypow - 1; p++)
                            ry_prev *= r_unit[k * 3 + 1];
                        der1 += rb_val * rx_pow * ypow * ry_prev * rz_pow / r_abs[k];
                    }
                    if (zpow > 0)
                    {
                        double rz_prev = 1.0;
                        for (int p = 0; p < zpow - 1; p++)
                            rz_prev *= r_unit[k * 3 + 2];
                        der2 += rb_val * rx_pow * ry_pow * zpow * rz_prev / r_abs[k];
                    }

                    int base_rc = ((((i_aib * species_count + jtype) * rfc + mu) * rbs + i_rb) * n_neighbors + k) * 3;
                    moment_jac_rc[base_rc + 0] += der0;
                    moment_jac_rc[base_rc + 1] += der1;
                    moment_jac_rc[base_rc + 2] += der2;
                }
            }
        }
    }
}

/* ==========================================================================
 * Contract moments through contraction operations
 * Computes all alpha_moments from basic moments
 * ========================================================================== */
static inline void contract_moments_forward(
    double *moments,
    const int *alpha_index_times,
    int n_times)
{
    for (int i_t = 0; i_t < n_times; i_t++)
    {
        int i1 = alpha_index_times[i_t * 4 + 0];
        int i2 = alpha_index_times[i_t * 4 + 1];
        int mult = alpha_index_times[i_t * 4 + 2];
        int i3 = alpha_index_times[i_t * 4 + 3];
        moments[i3] += mult * moments[i1] * moments[i2];
    }
}

/* ==========================================================================
 * Contract moment jacobians through contraction operations
 * ========================================================================== */
static inline void contract_moment_jacobians_forward(
    const double *moments,
    double *moment_jacobians,
    const int *alpha_index_times,
    int n_times,
    int n_neighbors)
{
    for (int i_t = 0; i_t < n_times; i_t++)
    {
        int i1 = alpha_index_times[i_t * 4 + 0];
        int i2 = alpha_index_times[i_t * 4 + 1];
        int mult = alpha_index_times[i_t * 4 + 2];
        int i3 = alpha_index_times[i_t * 4 + 3];

        /* For each neighbor slot k and component xyz */
        for (int k = 0; k < n_neighbors; k++)
        {
            for (int xyz = 0; xyz < 3; xyz++)
            {
                int idx_i1 = (i1 * n_neighbors + k) * 3 + xyz;
                int idx_i2 = (i2 * n_neighbors + k) * 3 + xyz;
                int idx_i3 = (i3 * n_neighbors + k) * 3 + xyz;

                /* d(m_i3)/d(r) += mult * (m_i1 * d(m_i2)/d(r) + m_i2 * d(m_i1)/d(r)) */
                moment_jacobians[idx_i3] += mult * (moments[i1] * moment_jacobians[idx_i2] +
                                                    moments[i2] * moment_jacobians[idx_i1]);
            }
        }
    }
}

/* ==========================================================================
 * Compute dedmb and dgdmb (backprop through moment contractions)
 * ========================================================================== */
static inline void compute_dedmb_dgdmb(
    int n_neighbors,
    const int *alpha_index_times,
    int n_times,
    const int *alpha_moment_mapping,
    int n_alpha_scalar,
    const double *moment_coeffs,
    const double *moment_values,
    const double *moment_jac_rs,
    double *dedmb,
    double *dgdmb)
{

    for (int i = 0; i < n_alpha_scalar; i++)
    {
        dedmb[alpha_moment_mapping[i]] = moment_coeffs[i];
    }

    for (int it = n_times - 1; it >= 0; it--)
    {
        int i1 = alpha_index_times[it * 4 + 0];
        int i2 = alpha_index_times[it * 4 + 1];
        int mult = alpha_index_times[it * 4 + 2];
        int i3 = alpha_index_times[it * 4 + 3];

        dedmb[i1] += mult * dedmb[i3] * moment_values[i2];
        dedmb[i2] += mult * dedmb[i3] * moment_values[i1];
    }

    for (int it = 0; it < n_times; it++)
    {
        int i1 = alpha_index_times[it * 4 + 0];
        int i2 = alpha_index_times[it * 4 + 1];
        int mult = alpha_index_times[it * 4 + 2];
        int i3 = alpha_index_times[it * 4 + 3];

        for (int k = 0; k < n_neighbors; k++)
        {
            for (int ixyz = 0; ixyz < 3; ixyz++)
            {
                dgdmb[(i1 * n_neighbors + k) * 3 + ixyz] +=
                    mult * dedmb[i3] * moment_jac_rs[(i2 * n_neighbors + k) * 3 + ixyz];
                dgdmb[(i2 * n_neighbors + k) * 3 + ixyz] +=
                    mult * dedmb[i3] * moment_jac_rs[(i1 * n_neighbors + k) * 3 + ixyz];
            }
        }
    }

    for (int it = n_times - 1; it >= 0; it--)
    {
        int i1 = alpha_index_times[it * 4 + 0];
        int i2 = alpha_index_times[it * 4 + 1];
        int mult = alpha_index_times[it * 4 + 2];
        int i3 = alpha_index_times[it * 4 + 3];

        for (int k = 0; k < n_neighbors; k++)
        {
            for (int ixyz = 0; ixyz < 3; ixyz++)
            {
                dgdmb[(i1 * n_neighbors + k) * 3 + ixyz] +=
                    mult * dgdmb[(i3 * n_neighbors + k) * 3 + ixyz] * moment_values[i2];
                dgdmb[(i2 * n_neighbors + k) * 3 + ixyz] +=
                    mult * dgdmb[(i3 * n_neighbors + k) * 3 + ixyz] * moment_values[i1];
            }
        }
    }
}

/* ==========================================================================
 * Accumulate dgdcs from moment_jac_cs/moment_jac_rc and dedmb/dgdmb
 * ========================================================================== */
static inline void accumulate_mbd_dgdcs_dsdcs(
    int i,
    int itype,
    int n_atoms,
    int n_neighbors,
    const int *js_i,
    const double *rs_i,
    int n_basic,
    int species_count,
    int radial_funcs_count,
    int radial_basis_size,
    const double *moment_jac_cs,
    const double *moment_jac_rc,
    const double *dedmb,
    const double *dgdmb,
    double *dgdcs,
    double *dsdcs)
{
    int rfc = radial_funcs_count;
    int rbs = radial_basis_size;

    int tmp_size = species_count * rfc * rbs * n_neighbors * 3;
    double *tmp_dgdcs = (double *)calloc(tmp_size, sizeof(double));

    for (int iamc = 0; iamc < n_basic; iamc++)
    {
        double v1 = dedmb[iamc];

        for (int ispc = 0; ispc < species_count; ispc++)
        {
            for (int irf = 0; irf < rfc; irf++)
            {
                for (int irb = 0; irb < rbs; irb++)
                {
                    int base_tmp = (((ispc * rfc + irf) * rbs + irb) * n_neighbors) * 3;
                    int base_rc = ((((iamc * species_count + ispc) * rfc + irf) * rbs + irb) * n_neighbors) * 3;

                    for (int j = 0; j < n_neighbors; j++)
                    {
                        for (int ixyz = 0; ixyz < 3; ixyz++)
                        {
                            tmp_dgdcs[base_tmp + j * 3 + ixyz] +=
                                moment_jac_rc[base_rc + j * 3 + ixyz] * v1;
                        }
                    }
                }
            }
        }
    }

    for (int iamc = 0; iamc < n_basic; iamc++)
    {
        for (int ispc = 0; ispc < species_count; ispc++)
        {
            for (int irf = 0; irf < rfc; irf++)
            {
                for (int irb = 0; irb < rbs; irb++)
                {
                    int idx_cs = ((iamc * species_count + ispc) * rfc + irf) * rbs + irb;
                    double v0 = moment_jac_cs[idx_cs];

                    if (v0 == 0.0)
                        continue;

                    int base_tmp = (((ispc * rfc + irf) * rbs + irb) * n_neighbors) * 3;
                    int base_mb = (iamc * n_neighbors) * 3;

                    for (int j = 0; j < n_neighbors; j++)
                    {
                        for (int ixyz = 0; ixyz < 3; ixyz++)
                        {
                            tmp_dgdcs[base_tmp + j * 3 + ixyz] +=
                                v0 * dgdmb[base_mb + j * 3 + ixyz];
                        }
                    }
                }
            }
        }
    }

    for (int ispc = 0; ispc < species_count; ispc++)
    {
        for (int irf = 0; irf < rfc; irf++)
        {
            for (int irb = 0; irb < rbs; irb++)
            {
                int base_tmp = (((ispc * rfc + irf) * rbs + irb) * n_neighbors) * 3;

                for (int k = 0; k < n_neighbors; k++)
                {
                    int j = js_i[k];

                    for (int ixyz = 0; ixyz < 3; ixyz++)
                    {
                        double v = tmp_dgdcs[base_tmp + k * 3 + ixyz];

                        int idx_i = (((((itype * species_count + ispc) * rfc + irf) * rbs + irb) * n_atoms + i) * 3 + ixyz);
                        dgdcs[idx_i] -= v;

                        if (j >= 0 && j < n_atoms)
                        {
                            int idx_j = (((((itype * species_count + ispc) * rfc + irf) * rbs + irb) * n_atoms + j) * 3 + ixyz);
                            dgdcs[idx_j] += v;
                        }
                    }

                    for (int ixyz0 = 0; ixyz0 < 3; ixyz0++)
                    {
                        for (int ixyz1 = 0; ixyz1 < 3; ixyz1++)
                        {
                            double v = rs_i[k * 3 + ixyz0] * tmp_dgdcs[base_tmp + k * 3 + ixyz1];
                            int idx_s = (((((itype * species_count + ispc) * rfc + irf) * rbs + irb) * 3 + ixyz0) * 3 + ixyz1);
                            dsdcs[idx_s] += v;
                        }
                    }
                }
            }
        }
    }

    free(tmp_dgdcs);
}

/* ==========================================================================
 * Accumulate mbd.dedcs from basic moment jacobians
 * ========================================================================== */
static inline void accumulate_mbd_dedcs(
    int itype,
    int n_basic,
    int species_count,
    int radial_funcs_count,
    int radial_basis_size,
    const double *moment_jac_cs,
    const double *dedmb,
    double *dedcs)
{
    int rbs = radial_basis_size;
    int rfc = radial_funcs_count;

    for (int iamc = 0; iamc < n_basic; iamc++)
    {
        double v1 = dedmb[iamc];

        for (int jtype_idx = 0; jtype_idx < species_count; jtype_idx++)
        {
            for (int irf = 0; irf < rfc; irf++)
            {
                for (int irb = 0; irb < rbs; irb++)
                {
                    int mjac_idx = ((iamc * species_count + jtype_idx) * rfc + irf) * rbs + irb;
                    double mjac_val = moment_jac_cs[mjac_idx];

                    int dedcs_idx = ((itype * species_count + jtype_idx) * rfc + irf) * rbs + irb;
                    dedcs[dedcs_idx] += mjac_val * v1;
                }
            }
        }
    }
}

/* ==========================================================================
 * Accumulate mapped moments into mbd_vatoms
 * ========================================================================== */
static inline void accumulate_mbd_vatoms(
    int i,
    int n_atoms,
    int n_alpha_scalar,
    const int *alpha_moment_mapping,
    const double *m,
    double *mbd_vatoms)
{
    for (int iamc = 0; iamc < n_alpha_scalar; iamc++)
    {
        mbd_vatoms[iamc * n_atoms + i] += m[alpha_moment_mapping[iamc]];
    }
}

/* ==========================================================================
 * Accumulate mbd.dbdris and mbd.dbdeps from moment jacobians
 * ========================================================================== */
static inline void accumulate_mbd_dbdris_dbdeps(
    int i,
    int n_atoms,
    int n_neighbors,
    const int *js_i,
    const double *rs_i,
    int n_alpha_scalar,
    const int *alpha_moment_mapping,
    const double *m_jac,
    double *mbd_dbdris,
    double *mbd_dbdeps)
{
    for (int iamc = 0; iamc < n_alpha_scalar; iamc++)
    {
        int alpha_idx = alpha_moment_mapping[iamc];

        for (int k = 0; k < n_neighbors; k++)
        {
            int j = js_i[k];

            for (int ixyz = 0; ixyz < 3; ixyz++)
            {
                double jac_contrib = m_jac[(alpha_idx * n_neighbors + k) * 3 + ixyz];

                mbd_dbdris[iamc * n_atoms * 3 + i * 3 + ixyz] -= jac_contrib;

                if (j >= 0 && j < n_atoms)
                {
                    mbd_dbdris[iamc * n_atoms * 3 + j * 3 + ixyz] += jac_contrib;
                }

                for (int ixyz1 = 0; ixyz1 < 3; ixyz1++)
                {
                    double stress_contrib = jac_contrib * rs_i[k * 3 + ixyz1];
                    mbd_dbdeps[iamc * 3 * 3 + ixyz * 3 + ixyz1] += stress_contrib;
                }
            }
        }
    }
}

/* ==========================================================================
 * Backward jacobian accumulation for run mode.  Assumes moment_values has
 * already been forward-contracted.  Seeds dedmb from moment_coeffs at
 * scalar-moment indices, back-propagates through contractions, then accumulates
 * gradient from the uncontracted basic jacobian.
 * ========================================================================== */
static inline void contract_moment_jacobians_backwards(
    int n_basic,
    int alpha_moments_count,
    int n_alpha_scalar,
    const int *alpha_moment_mapping,
    const int *alpha_index_times,
    int n_times,
    const double *moment_coeffs,
    const double *moment_values,
    const double *moment_jac,
    int n_neighbors,
    double *grad)
{
    double *dedmb = (double *)calloc(alpha_moments_count, sizeof(double));
    for (int i = 0; i < n_alpha_scalar; i++)
    {
        dedmb[alpha_moment_mapping[i]] = moment_coeffs[i];
    }

    for (int it = n_times - 1; it >= 0; it--)
    {
        int i1 = alpha_index_times[it * 4 + 0];
        int i2 = alpha_index_times[it * 4 + 1];
        int mult = alpha_index_times[it * 4 + 2];
        int i3 = alpha_index_times[it * 4 + 3];

        dedmb[i2] += dedmb[i3] * mult * moment_values[i1];
        dedmb[i1] += dedmb[i3] * mult * moment_values[i2];
    }

    for (int aib = 0; aib < n_basic; aib++)
    {
        double der = dedmb[aib];
        for (int j = 0; j < n_neighbors; j++)
        {
            int base = (aib * n_neighbors + j) * 3;
            grad[j * 3 + 0] += der * moment_jac[base + 0];
            grad[j * 3 + 1] += der * moment_jac[base + 1];
            grad[j * 3 + 2] += der * moment_jac[base + 2];
        }
    }

    free(dedmb);
}

/* ==========================================================================
 * Helper: Calculate distances and unit vectors
 * ========================================================================== */
static inline void calc_distances_and_unit_vectors(
    int n_neighbors,
    const double *rs_i,
    double *r_abs,
    double *r_unit)
{
    for (int k = 0; k < n_neighbors; k++)
    {
        double rx = rs_i[k * 3 + 0];
        double ry = rs_i[k * 3 + 1];
        double rz = rs_i[k * 3 + 2];
        r_abs[k] = sqrt(rx * rx + ry * ry + rz * rz);

        if (r_abs[k] > 1e-10)
        {
            r_unit[k * 3 + 0] = rx / r_abs[k];
            r_unit[k * 3 + 1] = ry / r_abs[k];
            r_unit[k * 3 + 2] = rz / r_abs[k];
        }
        else
        {
            r_unit[k * 3 + 0] = 0.0;
            r_unit[k * 3 + 1] = 0.0;
            r_unit[k * 3 + 2] = 0.0;
        }
    }
}

/* ==========================================================================
 * Helper: Store radial basis values and derivatives for training mode
 * ========================================================================== */
static inline void store_radial_basis_train(
    int i, int itype, int n_atoms, int n_neighbors, int species_count, int rbs,
    const int *jtype_i,
    const int *js_i,
    const double *r_abs, const double *r_unit, const double *r_i,
    const double *radial_basis, const double *drb_drs,
    double *rbd_values,
    double *rbd_dqdris,
    double *rbd_dqdeps)
{
    for (int irb = 0; irb < rbs; irb++)
    {
        for (int k = 0; k < n_neighbors; k++)
        {
            int jtype = jtype_i[k];
            int j = js_i[k];
            double rb_val = radial_basis[irb * n_neighbors + k];
            double rb_deriv = drb_drs[irb * n_neighbors + k];

            /* Store radial basis values: rbd_values[itype][jtype][irb] */
            int idx_val = ((itype * species_count + jtype) * rbs + irb);
            rbd_values[idx_val] += rb_val;

            if (r_abs[k] > 1e-10)
            {
                /* Store derivatives w.r.t. atomic positions */
                for (int xyz = 0; xyz < 3; xyz++)
                {
                    double tmp = rb_deriv * r_unit[k * 3 + xyz];

                    /* dqdris[itype][jtype][irb][atom_i][xyz] -= tmp */
                    int idx_i = ((itype * species_count + jtype) * rbs + irb) * n_atoms * 3 + i * 3 + xyz;
                    rbd_dqdris[idx_i] -= tmp;

                    /* dqdris[itype][jtype][irb][atom_j][xyz] += tmp (if j is valid) */
                    if (j >= 0 && j < n_atoms)
                    {
                        int idx_j = ((itype * species_count + jtype) * rbs + irb) * n_atoms * 3 + j * 3 + xyz;
                        rbd_dqdris[idx_j] += tmp;
                    }

                    /* Store derivatives w.r.t. strain: dqdeps[itype][jtype][irb][xyz1][xyz2] */
                    for (int xyz2 = 0; xyz2 < 3; xyz2++)
                    {
                        double tmp_eps = tmp * r_i[k * 3 + xyz2];
                        int idx_eps = ((itype * species_count + jtype) * rbs + irb) * 3 * 3 + xyz * 3 + xyz2;
                        rbd_dqdeps[idx_eps] += tmp_eps;
                    }
                }
            }
        }
    }
}

#endif /* MTP_CEXT_KERNELS_H */
