#ifndef MMTP_CEXT_KERNELS_H
#define MMTP_CEXT_KERNELS_H

#include <math.h>
#include <stdlib.h>

#include "mtp_cext_kernels.h"

/* ==========================================================================
 * Helper: Chebyshev polynomials without smoothing (for magnetic basis)
 * ========================================================================== */
static inline void chebyshev_plain(
    double x, int n_basis,
    double min_val, double max_val,
    double *basis_vals, double *basis_derivs)
{
    double r_scaled = (2.0 * x - (min_val + max_val)) / (max_val - min_val);
    double r_deriv = 2.0 / (max_val - min_val);

    basis_vals[0] = 1.0;
    basis_derivs[0] = 0.0;

    if (n_basis > 1)
    {
        basis_vals[1] = r_scaled;
        basis_derivs[1] = r_deriv;
    }

    for (int i = 2; i < n_basis; i++)
    {
        basis_vals[i] = 2.0 * r_scaled * basis_vals[i - 1] - basis_vals[i - 2];
        basis_derivs[i] = 2.0 * (r_scaled * basis_derivs[i - 1] + r_deriv * basis_vals[i - 1]) - basis_derivs[i - 2];
    }
}

/* ==========================================================================
 * Helper: Calculate radial + magnetic basis values and derivatives
 * Layout: [nrb][n_neighbors] where nrb = rbs * mbs^2
 * ========================================================================== */
static inline void calc_radial_and_mag_basis(
    int n_neighbors,
    const double *r_abs,
    const double *ms,
    int radial_basis_size,
    int mag_basis_size,
    double scaling, double min_dist, double max_dist,
    double mi, double min_mag, double max_mag,
    double *basis, double *dbdrs, double *dbdmis, double *dbdmjs)
{
    int rbs = radial_basis_size;
    int mbs = mag_basis_size;
    int nrb = rbs * mbs * mbs;

    double *tmp_rb_vals = (double *)malloc(rbs * sizeof(double));
    double *tmp_drb_drs = (double *)malloc(rbs * sizeof(double));
    double *tmp_mb1_vals = (double *)malloc(mbs * sizeof(double));
    double *tmp_mb1_ders = (double *)malloc(mbs * sizeof(double));
    double *tmp_mb2_vals = (double *)malloc(mbs * sizeof(double));
    double *tmp_mb2_ders = (double *)malloc(mbs * sizeof(double));

    /* Magnetic basis for central atom (mi) is constant across neighbors */
    chebyshev_plain(mi, mbs, min_mag, max_mag, tmp_mb1_vals, tmp_mb1_ders);

    for (int j = 0; j < n_neighbors; j++)
    {
        /* Radial basis with smoothing */
        chebyshev_basis(r_abs[j], rbs, scaling, min_dist, max_dist,
                        tmp_rb_vals, tmp_drb_drs);

        /* Magnetic basis for neighbor j */
        chebyshev_plain(ms[j], mbs, min_mag, max_mag, tmp_mb2_vals, tmp_mb2_ders);

        for (int irb = 0; irb < rbs; irb++)
        {
            double rb_val = tmp_rb_vals[irb];
            double rb_der = tmp_drb_drs[irb];

            for (int imb1 = 0; imb1 < mbs; imb1++)
            {
                double bm1 = tmp_mb1_vals[imb1];
                double dm1 = tmp_mb1_ders[imb1];

                for (int imb2 = 0; imb2 < mbs; imb2++)
                {
                    double bm2 = tmp_mb2_vals[imb2];
                    double dm2 = tmp_mb2_ders[imb2];

                    int ind = irb * mbs * mbs + imb1 * mbs + imb2;
                    int idx = ind * n_neighbors + j;

                    basis[idx] = rb_val * bm1 * bm2;
                    dbdrs[idx] = rb_der * bm1 * bm2;
                    if (dbdmis)
                        dbdmis[idx] = rb_val * dm1 * bm2;
                    if (dbdmjs)
                        dbdmjs[idx] = rb_val * bm1 * dm2;
                }
            }
        }
    }

    free(tmp_rb_vals);
    free(tmp_drb_drs);
    free(tmp_mb1_vals);
    free(tmp_mb1_ders);
    free(tmp_mb2_vals);
    free(tmp_mb2_ders);
}

/* ==========================================================================
 * Helper: Calculate basic magnetic moments and jacobians
 * ========================================================================== */
static inline void calc_mag_basic_moments(
    int n_neighbors,
    const double *r_abs,
    const double *r_unit,
    const double *rf_vals,
    const double *drf_drs,
    const double *drf_dmis,
    const double *drf_dmjs,
    const int *alpha_index_basic,
    int n_basic,
    double *moment_values,
    double *moment_jac_rs,
    double *moment_jac_mis,
    double *moment_jac_mjs)
{
    for (int i_aib = 0; i_aib < n_basic; i_aib++)
    {
        int mu = alpha_index_basic[i_aib * 4 + 0];
        int xpow = alpha_index_basic[i_aib * 4 + 1];
        int ypow = alpha_index_basic[i_aib * 4 + 2];
        int zpow = alpha_index_basic[i_aib * 4 + 3];
        int xyzpow = xpow + ypow + zpow;

        for (int k = 0; k < n_neighbors; k++)
        {
            double rf = rf_vals[mu * n_neighbors + k];
            double drf_dr = drf_drs[mu * n_neighbors + k];

            double rx_pow = 1.0, ry_pow = 1.0, rz_pow = 1.0;
            for (int p = 0; p < xpow; p++)
                rx_pow *= r_unit[k * 3 + 0];
            for (int p = 0; p < ypow; p++)
                ry_pow *= r_unit[k * 3 + 1];
            for (int p = 0; p < zpow; p++)
                rz_pow *= r_unit[k * 3 + 2];

            double angular = rx_pow * ry_pow * rz_pow;

            moment_values[i_aib] += rf * angular;
            moment_jac_mis[i_aib * n_neighbors + k] = drf_dmis[mu * n_neighbors + k] * angular;
            moment_jac_mjs[i_aib * n_neighbors + k] = drf_dmjs[mu * n_neighbors + k] * angular;

            if (r_abs[k] > 1e-10)
            {
                double radial_jac = angular * (drf_dr - xyzpow * rf / r_abs[k]);
                moment_jac_rs[(i_aib * n_neighbors + k) * 3 + 0] = r_unit[k * 3 + 0] * radial_jac;
                moment_jac_rs[(i_aib * n_neighbors + k) * 3 + 1] = r_unit[k * 3 + 1] * radial_jac;
                moment_jac_rs[(i_aib * n_neighbors + k) * 3 + 2] = r_unit[k * 3 + 2] * radial_jac;

                if (xpow > 0)
                {
                    double rx_prev = 1.0;
                    for (int p = 0; p < xpow - 1; p++)
                        rx_prev *= r_unit[k * 3 + 0];
                    moment_jac_rs[(i_aib * n_neighbors + k) * 3 + 0] += rf * xpow * rx_prev * ry_pow * rz_pow / r_abs[k];
                }
                if (ypow > 0)
                {
                    double ry_prev = 1.0;
                    for (int p = 0; p < ypow - 1; p++)
                        ry_prev *= r_unit[k * 3 + 1];
                    moment_jac_rs[(i_aib * n_neighbors + k) * 3 + 1] += rf * rx_pow * ypow * ry_prev * rz_pow / r_abs[k];
                }
                if (zpow > 0)
                {
                    double rz_prev = 1.0;
                    for (int p = 0; p < zpow - 1; p++)
                        rz_prev *= r_unit[k * 3 + 2];
                    moment_jac_rs[(i_aib * n_neighbors + k) * 3 + 2] += rf * rx_pow * ry_pow * zpow * rz_prev / r_abs[k];
                }
            }
        }
    }
}

/* ==========================================================================
 * Helper: Contract moments and compute gradients (backward)
 * ========================================================================== */
static inline void mag_contract_moments_backwards(
    int n_basic,
    int alpha_moments_count,
    int n_alpha_scalar,
    const int *alpha_moment_mapping,
    const int *alpha_index_times,
    int n_times,
    const double *moment_coeffs,
    double *moment_values,
    const double *moment_jac_rs,
    const double *moment_jac_mis,
    const double *moment_jac_mjs,
    int n_neighbors,
    double *grad,
    double *grad_mag_i,
    double *grad_mag_j)
{
    /* Forward contraction of moments */
    contract_moments_forward(moment_values, alpha_index_times, n_times);

    double *tmp_moment_ders = (double *)calloc(alpha_moments_count, sizeof(double));
    for (int i = 0; i < n_alpha_scalar; i++)
    {
        tmp_moment_ders[alpha_moment_mapping[i]] = moment_coeffs[i];
    }

    for (int it = n_times - 1; it >= 0; it--)
    {
        int i1 = alpha_index_times[it * 4 + 0];
        int i2 = alpha_index_times[it * 4 + 1];
        int mult = alpha_index_times[it * 4 + 2];
        int i3 = alpha_index_times[it * 4 + 3];

        tmp_moment_ders[i2] += tmp_moment_ders[i3] * mult * moment_values[i1];
        tmp_moment_ders[i1] += tmp_moment_ders[i3] * mult * moment_values[i2];
    }

    for (int aib = 0; aib < n_basic; aib++)
    {
        double der = tmp_moment_ders[aib];
        for (int j = 0; j < n_neighbors; j++)
        {
            grad_mag_i[j] += der * moment_jac_mis[aib * n_neighbors + j];
            grad_mag_j[j] += der * moment_jac_mjs[aib * n_neighbors + j];

            int base = (aib * n_neighbors + j) * 3;
            grad[j * 3 + 0] += der * moment_jac_rs[base + 0];
            grad[j * 3 + 1] += der * moment_jac_rs[base + 1];
            grad[j * 3 + 2] += der * moment_jac_rs[base + 2];
        }
    }

    free(tmp_moment_ders);
}

/* ==========================================================================
 * Helper: Store magnetic radial basis values and derivatives for train_mgrad
 * ========================================================================== */
static inline void store_mag_radial_basis_train(
    int i, int itype, int n_atoms, int n_neighbors, int species_count, int nrb,
    const int *jtype_i,
    const int *js_i,
    const double *r_abs, const double *r_unit, const double *r_i,
    const double *basis, const double *dbdrs,
    const double *dbdmis, const double *dbdmjs,
    double *rbd_values,
    double *rbd_dqdris,
    double *rbd_dqdmis,
    double *rbd_dqdeps)
{
    for (int irb = 0; irb < nrb; irb++)
    {
        for (int k = 0; k < n_neighbors; k++)
        {
            int jtype = jtype_i[k];
            int j = js_i[k];
            double rb_val = basis[irb * n_neighbors + k];
            double rb_deriv = dbdrs[irb * n_neighbors + k];
            double rb_dmi = dbdmis[irb * n_neighbors + k];
            double rb_dmj = dbdmjs[irb * n_neighbors + k];

            int idx_val = ((itype * species_count + jtype) * nrb + irb);
            rbd_values[idx_val] += rb_val;

            int idx_mi = ((itype * species_count + jtype) * nrb + irb) * n_atoms + i;
            rbd_dqdmis[idx_mi] += rb_dmi;

            if (j >= 0 && j < n_atoms)
            {
                int idx_mj = ((itype * species_count + jtype) * nrb + irb) * n_atoms + j;
                rbd_dqdmis[idx_mj] += rb_dmj;
            }

            if (r_abs[k] > 1e-10)
            {
                for (int xyz = 0; xyz < 3; xyz++)
                {
                    double tmp = rb_deriv * r_unit[k * 3 + xyz];

                    int idx_i = ((itype * species_count + jtype) * nrb + irb) * n_atoms * 3 + i * 3 + xyz;
                    rbd_dqdris[idx_i] -= tmp;

                    if (j >= 0 && j < n_atoms)
                    {
                        int idx_j = ((itype * species_count + jtype) * nrb + irb) * n_atoms * 3 + j * 3 + xyz;
                        rbd_dqdris[idx_j] += tmp;
                    }

                    for (int xyz2 = 0; xyz2 < 3; xyz2++)
                    {
                        double tmp_eps = tmp * r_i[k * 3 + xyz2];
                        int idx_eps = ((itype * species_count + jtype) * nrb + irb) * 3 * 3 + xyz * 3 + xyz2;
                        rbd_dqdeps[idx_eps] += tmp_eps;
                    }
                }
            }
        }
    }
}

/* ==========================================================================
 * Helper: Contract magnetic moment jacobians (w.r.t. magnetic moments)
 * ========================================================================== */
static inline void contract_mag_moment_jacobians_forward(
    const double *moments,
    double *moment_jac_mis,
    double *moment_jac_mjs,
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

        for (int k = 0; k < n_neighbors; k++)
        {
            int idx_i1 = i1 * n_neighbors + k;
            int idx_i2 = i2 * n_neighbors + k;
            int idx_i3 = i3 * n_neighbors + k;

            moment_jac_mis[idx_i3] += mult * (moment_jac_mis[idx_i1] * moments[i2] +
                                              moments[i1] * moment_jac_mis[idx_i2]);
            moment_jac_mjs[idx_i3] += mult * (moment_jac_mjs[idx_i1] * moments[i2] +
                                              moments[i1] * moment_jac_mjs[idx_i2]);
        }
    }
}

/* ==========================================================================
 * Helper: Compute dedmb, dgdmb and magnetic derivatives dgmidmb/dgmjdmb
 * ========================================================================== */
static inline void compute_dedmb_dgdmb_dgmdmb(
    int n_neighbors,
    const int *alpha_index_times,
    int n_times,
    const int *alpha_moment_mapping,
    int n_alpha_scalar,
    const double *moment_coeffs,
    const double *moment_values,
    const double *moment_jac_rs,
    const double *moment_jac_mis,
    const double *moment_jac_mjs,
    double *dedmb,
    double *dgdmb,
    double *dgmidmb,
    double *dgmjdmb)
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

        for (int j = 0; j < n_neighbors; j++)
        {
            dgmidmb[i1 * n_neighbors + j] += mult * dedmb[i3] * moment_jac_mis[i2 * n_neighbors + j];
            dgmidmb[i2 * n_neighbors + j] += mult * dedmb[i3] * moment_jac_mis[i1 * n_neighbors + j];
            dgmjdmb[i1 * n_neighbors + j] += mult * dedmb[i3] * moment_jac_mjs[i2 * n_neighbors + j];
            dgmjdmb[i2 * n_neighbors + j] += mult * dedmb[i3] * moment_jac_mjs[i1 * n_neighbors + j];

            int base_i1 = (i1 * n_neighbors + j) * 3;
            int base_i2 = (i2 * n_neighbors + j) * 3;
            for (int ixyz = 0; ixyz < 3; ixyz++)
            {
                dgdmb[base_i1 + ixyz] += mult * dedmb[i3] * moment_jac_rs[base_i2 + ixyz];
                dgdmb[base_i2 + ixyz] += mult * dedmb[i3] * moment_jac_rs[base_i1 + ixyz];
            }
        }
    }

    for (int it = n_times - 1; it >= 0; it--)
    {
        int i1 = alpha_index_times[it * 4 + 0];
        int i2 = alpha_index_times[it * 4 + 1];
        int mult = alpha_index_times[it * 4 + 2];
        int i3 = alpha_index_times[it * 4 + 3];

        for (int j = 0; j < n_neighbors; j++)
        {
            dgmidmb[i1 * n_neighbors + j] += mult * dgmidmb[i3 * n_neighbors + j] * moment_values[i2];
            dgmidmb[i2 * n_neighbors + j] += mult * dgmidmb[i3 * n_neighbors + j] * moment_values[i1];
            dgmjdmb[i1 * n_neighbors + j] += mult * dgmjdmb[i3 * n_neighbors + j] * moment_values[i2];
            dgmjdmb[i2 * n_neighbors + j] += mult * dgmjdmb[i3 * n_neighbors + j] * moment_values[i1];

            int base_i1 = (i1 * n_neighbors + j) * 3;
            int base_i2 = (i2 * n_neighbors + j) * 3;
            for (int ixyz = 0; ixyz < 3; ixyz++)
            {
                dgdmb[base_i1 + ixyz] += mult * dgdmb[(i3 * n_neighbors + j) * 3 + ixyz] * moment_values[i2];
                dgdmb[base_i2 + ixyz] += mult * dgdmb[(i3 * n_neighbors + j) * 3 + ixyz] * moment_values[i1];
            }
        }
    }
}

/* ==========================================================================
 * Helper: Accumulate mbd.dbdmis for train_mgrad
 * ========================================================================== */
static inline void accumulate_mbd_dbdmis(
    int i,
    int n_atoms,
    int n_neighbors,
    const int *js_i,
    int n_alpha_scalar,
    const int *alpha_moment_mapping,
    const double *moment_jac_mis,
    const double *moment_jac_mjs,
    double *mbd_dbdmis)
{
    for (int iamc = 0; iamc < n_alpha_scalar; iamc++)
    {
        int alpha_idx = alpha_moment_mapping[iamc];

        for (int k = 0; k < n_neighbors; k++)
        {
            int j = js_i[k];
            double v_i = moment_jac_mis[alpha_idx * n_neighbors + k];
            double v_j = moment_jac_mjs[alpha_idx * n_neighbors + k];

            mbd_dbdmis[iamc * n_atoms + i] += v_i;
            if (j >= 0 && j < n_atoms)
            {
                mbd_dbdmis[iamc * n_atoms + j] += v_j;
            }
        }
    }
}

/* ==========================================================================
 * Helper: Accumulate mbd.dgmdcs for train_mgrad
 * ========================================================================== */
static inline void accumulate_mbd_dgmdcs(
    int i,
    int itype,
    int n_atoms,
    int n_neighbors,
    const int *js_i,
    int n_basic,
    int species_count,
    int radial_funcs_count,
    int radial_basis_size,
    const double *moment_jac_cs,
    const double *moment_jac_mic,
    const double *moment_jac_mjc,
    const double *dedmb,
    const double *dgmidmb,
    const double *dgmjdmb,
    double *mbd_dgmdcs)
{
    int rfc = radial_funcs_count;
    int rbs = radial_basis_size;

    for (int iamc = 0; iamc < n_basic; iamc++)
    {
        double v1 = dedmb[iamc];

        for (int ispc = 0; ispc < species_count; ispc++)
        {
            for (int irf = 0; irf < rfc; irf++)
            {
                for (int irb = 0; irb < rbs; irb++)
                {
                    int idx_cs = ((iamc * species_count + ispc) * rfc + irf) * rbs + irb;
                    double v3 = moment_jac_cs[idx_cs];

                    if (v1 == 0.0 && v3 == 0.0)
                        continue;

                    int base_mic = (((iamc * species_count + ispc) * rfc + irf) * rbs + irb) * n_neighbors;
                    int base_atom = ((((itype * species_count + ispc) * rfc + irf) * rbs + irb) * n_atoms);

                    for (int k = 0; k < n_neighbors; k++)
                    {
                        int j = js_i[k];
                        double vmi = moment_jac_mic[base_mic + k] * v1 + v3 * dgmidmb[iamc * n_neighbors + k];
                        double vmj = moment_jac_mjc[base_mic + k] * v1 + v3 * dgmjdmb[iamc * n_neighbors + k];

                        mbd_dgmdcs[base_atom + i] += vmi;
                        if (j >= 0 && j < n_atoms)
                        {
                            mbd_dgmdcs[base_atom + j] += vmj;
                        }
                    }
                }
            }
        }
    }
}

/* ==========================================================================
 * Helper: Compute jacobians of basic magnetic moments w.r.t. radial coefficients
 * ========================================================================== */
static inline void calc_mag_basic_moments_jac_radial_coeffs(
    int n_neighbors,
    const double *r_abs,
    const double *r_unit,
    const double *rb_vals,
    const double *rb_derivs,
    const double *drb_dmis,
    const double *drb_dmjs,
    const int *alpha_index_basic,
    int n_basic,
    int species_count,
    const int *jtypes,
    int radial_funcs_count,
    int nrb,
    double *moment_jac_cs,
    double *moment_jac_rc,
    double *moment_jac_mic,
    double *moment_jac_mjc)
{
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

            double rx_pow = 1.0, ry_pow = 1.0, rz_pow = 1.0;
            for (int p = 0; p < xpow; p++)
                rx_pow *= r_unit[k * 3 + 0];
            for (int p = 0; p < ypow; p++)
                ry_pow *= r_unit[k * 3 + 1];
            for (int p = 0; p < zpow; p++)
                rz_pow *= r_unit[k * 3 + 2];

            double angular = rx_pow * ry_pow * rz_pow;

            for (int i_rb = 0; i_rb < nrb; i_rb++)
            {
                double rb_val = rb_vals[i_rb * n_neighbors + k];
                double rb_der = rb_derivs[i_rb * n_neighbors + k];
                double drb_dmi = drb_dmis[i_rb * n_neighbors + k];
                double drb_dmj = drb_dmjs[i_rb * n_neighbors + k];

                int idx_cs = ((i_aib * species_count + jtype) * rfc + mu) * nrb + i_rb;
                moment_jac_cs[idx_cs] += rb_val * angular;

                int idx_mic = ((i_aib * species_count + jtype) * rfc + mu) * nrb * n_neighbors + i_rb * n_neighbors + k;
                moment_jac_mic[idx_mic] += drb_dmi * angular;

                int idx_mjc = ((i_aib * species_count + jtype) * rfc + mu) * nrb * n_neighbors + i_rb * n_neighbors + k;
                moment_jac_mjc[idx_mjc] += drb_dmj * angular;

                if (r_abs[k] > 1e-10)
                {
                    double der0 = r_unit[k * 3 + 0] * angular * (rb_der - xyzpow * rb_val / r_abs[k]);
                    double der1 = r_unit[k * 3 + 1] * angular * (rb_der - xyzpow * rb_val / r_abs[k]);
                    double der2 = r_unit[k * 3 + 2] * angular * (rb_der - xyzpow * rb_val / r_abs[k]);

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

                    int base_rc = ((((i_aib * species_count + jtype) * rfc + mu) * nrb + i_rb) * n_neighbors + k) * 3;
                    moment_jac_rc[base_rc + 0] += der0;
                    moment_jac_rc[base_rc + 1] += der1;
                    moment_jac_rc[base_rc + 2] += der2;
                }
            }
        }
    }
}

#endif /* MMTP_CEXT_KERNELS_H */