#include "mmtp_cext.h"
#include "mmtp_cext_kernels.h"

/* ==========================================================================
 * Main run mode calculation
 * ========================================================================== */
void calc_mag_run(
    int n_atoms, int n_neighbors,
    const double *rs,
    const int *js,
    const double *ms,
    const int *itypes, const int *jtype,
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
    double *gradient,
    double *grad_mag_i,
    double *grad_mag_j,
    double *mbd_vatoms)
{
    int rbs = radial_basis_size;
    int mbs = mag_basis_size;
    int nrb = rbs * mbs * mbs;

    for (int i = 0; i < n_atoms; i++)
    {
        energies[i] = species_coeffs[itypes[i]];
    }

    memset(gradient, 0, n_atoms * n_neighbors * 3 * sizeof(double));
    memset(grad_mag_i, 0, n_atoms * n_neighbors * sizeof(double));
    memset(grad_mag_j, 0, n_atoms * n_neighbors * sizeof(double));

    for (int i = 0; i < n_atoms; i++)
    {
        int itype = itypes[i];
        const double *rs_i = rs + i * n_neighbors * 3;
        const int *js_i = js + i * n_neighbors;
        const int *jtype_i = jtype + i * n_neighbors;
        double *grad_i = gradient + i * n_neighbors * 3;
        double *grad_mi_i = grad_mag_i + i * n_neighbors;
        double *grad_mj_i = grad_mag_j + i * n_neighbors;

        double *r_abs = (double *)malloc(n_neighbors * sizeof(double));
        double *r_unit = (double *)malloc(n_neighbors * 3 * sizeof(double));

        calc_distances_and_unit_vectors(n_neighbors, rs_i, r_abs, r_unit);

        double *basis = (double *)malloc(4 * nrb * n_neighbors * sizeof(double));
        double *dbdrs = basis + nrb * n_neighbors;
        double *dbdmis = dbdrs + nrb * n_neighbors;
        double *dbdmjs = dbdmis + nrb * n_neighbors;

        double *ms_neighbors = (double *)malloc(n_neighbors * sizeof(double));
        for (int k = 0; k < n_neighbors; k++)
        {
            int j = js_i[k];
            ms_neighbors[k] = (j >= 0 && j < n_atoms) ? ms[j] : 0.0;
        }

        calc_radial_and_mag_basis(
            n_neighbors,
            r_abs,
            ms_neighbors,
            rbs,
            mbs,
            scaling,
            min_dist,
            max_dist,
            ms[i],
            min_mag,
            max_mag,
            basis,
            dbdrs,
            dbdmis,
            dbdmjs);

        double *radial_funcs = (double *)malloc(4 * radial_funcs_count * n_neighbors * sizeof(double));
        double *drf_drs = radial_funcs + radial_funcs_count * n_neighbors;
        double *drf_dmis = drf_drs + radial_funcs_count * n_neighbors;
        double *drf_dmjs = drf_dmis + radial_funcs_count * n_neighbors;

        sum_radial_basis(itype, jtype_i, n_neighbors,
                         basis, nrb,
                         radial_coeffs, species_count, radial_funcs_count,
                         radial_funcs);

        sum_radial_basis(itype, jtype_i, n_neighbors,
                         dbdrs, nrb,
                         radial_coeffs, species_count, radial_funcs_count,
                         drf_drs);

        sum_radial_basis(itype, jtype_i, n_neighbors,
                         dbdmis, nrb,
                         radial_coeffs, species_count, radial_funcs_count,
                         drf_dmis);

        sum_radial_basis(itype, jtype_i, n_neighbors,
                         dbdmjs, nrb,
                         radial_coeffs, species_count, radial_funcs_count,
                         drf_dmjs);

        double *moment_values = (double *)calloc(alpha_moments_count, sizeof(double));
        double *moment_jac_rs = (double *)calloc(alpha_moments_count * n_neighbors * 3, sizeof(double));
        double *moment_jac_mis = (double *)calloc(alpha_moments_count * n_neighbors, sizeof(double));
        double *moment_jac_mjs = (double *)calloc(alpha_moments_count * n_neighbors, sizeof(double));

        calc_mag_basic_moments(
            n_neighbors,
            r_abs,
            r_unit,
            radial_funcs,
            drf_drs,
            drf_dmis,
            drf_dmjs,
            alpha_index_basic,
            n_basic,
            moment_values,
            moment_jac_rs,
            moment_jac_mis,
            moment_jac_mjs);

        mag_contract_moments_backwards(
            n_basic,
            alpha_moments_count,
            n_alpha_scalar,
            alpha_moment_mapping,
            alpha_index_times,
            n_times,
            moment_coeffs,
            moment_values,
            moment_jac_rs,
            moment_jac_mis,
            moment_jac_mjs,
            n_neighbors,
            grad_i,
            grad_mi_i,
            grad_mj_i);

        for (int i_am = 0; i_am < n_alpha_scalar; i_am++)
        {
            int idx = alpha_moment_mapping[i_am];
            energies[i] += moment_coeffs[i_am] * moment_values[idx];
        }

        accumulate_mbd_vatoms(
            i,
            n_atoms,
            n_alpha_scalar,
            alpha_moment_mapping,
            moment_values,
            mbd_vatoms);

        free(r_abs);
        free(r_unit);
        free(basis);
        free(ms_neighbors);
        free(radial_funcs);
        free(moment_values);
        free(moment_jac_rs);
        free(moment_jac_mis);
        free(moment_jac_mjs);
    }
}

/* ==========================================================================
 * Calculate forces from gradients
 * ========================================================================== */
void calc_forces_from_gradient(
    const double *gradient,
    const int *js,
    int n_atoms, int n_neighbors,
    double *forces)
{
    memset(forces, 0, n_atoms * 3 * sizeof(double));

    for (int i = 0; i < n_atoms; i++)
    {
        for (int k = 0; k < n_neighbors; k++)
        {
            int j = js[i * n_neighbors + k];
            if (j < 0)
                continue;
            for (int xyz = 0; xyz < 3; xyz++)
            {
                double grad_val = gradient[(i * n_neighbors + k) * 3 + xyz];
                forces[i * 3 + xyz] += grad_val;
                forces[j * 3 + xyz] -= grad_val;
            }
        }
    }
}

/* ==========================================================================
 * Calculate magnetic gradients from per-pair gradients
 * ========================================================================== */
void calc_mgrad_from_gradient(
    const double *grad_mag_i,
    const double *grad_mag_j,
    const int *js,
    int n_atoms, int n_neighbors,
    double *mgrad)
{
    memset(mgrad, 0, n_atoms * sizeof(double));

    for (int i = 0; i < n_atoms; i++)
    {
        for (int k = 0; k < n_neighbors; k++)
        {
            int j = js[i * n_neighbors + k];
            if (j < 0)
                continue;
            double gmi = grad_mag_i[i * n_neighbors + k];
            double gmj = grad_mag_j[i * n_neighbors + k];
            mgrad[i] += gmi;
            mgrad[j] += gmj;
        }
    }
}
/* ==========================================================================
 * Training mode calculation
 * ========================================================================== */
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
    double *mbd_dsdcs)
{
    int rbs = radial_basis_size;
    int mbs = mag_basis_size;
    int nrb = rbs * mbs * mbs;

    for (int i = 0; i < n_atoms; i++)
    {
        energies[i] = species_coeffs[itypes[i]];
    }

    for (int i = 0; i < n_atoms; i++)
    {
        int itype = itypes[i];
        const double *rs_i = rs + i * n_neighbors * 3;
        const int *js_i = js + i * n_neighbors;
        const int *jtype_i = jtype + i * n_neighbors;

        double *r_abs = (double *)malloc(n_neighbors * sizeof(double));
        double *r_unit = (double *)malloc(n_neighbors * 3 * sizeof(double));

        calc_distances_and_unit_vectors(n_neighbors, rs_i, r_abs, r_unit);

        double *basis = (double *)malloc(2 * nrb * n_neighbors * sizeof(double));
        double *dbdrs = basis + nrb * n_neighbors;

        double *ms_neighbors = (double *)malloc(n_neighbors * sizeof(double));
        for (int k = 0; k < n_neighbors; k++)
        {
            int j = js_i[k];
            ms_neighbors[k] = (j >= 0 && j < n_atoms) ? ms[j] : 0.0;
        }

        calc_radial_and_mag_basis(
            n_neighbors,
            r_abs,
            ms_neighbors,
            rbs,
            mbs,
            scaling,
            min_dist,
            max_dist,
            ms[i],
            min_mag,
            max_mag,
            basis,
            dbdrs,
            NULL,
            NULL);

        store_radial_basis_train(
            i, itype, n_atoms, n_neighbors, species_count, nrb,
            jtype_i, js_i,
            r_abs, r_unit, rs_i,
            basis, dbdrs,
            rbd_values, rbd_dqdris, rbd_dqdeps);

        double *radial_funcs = (double *)malloc(2 * radial_funcs_count * n_neighbors * sizeof(double));
        double *drf_drs = radial_funcs + radial_funcs_count * n_neighbors;

        sum_radial_basis(itype, jtype_i, n_neighbors,
                         basis, nrb,
                         radial_coeffs, species_count, radial_funcs_count,
                         radial_funcs);

        sum_radial_basis(itype, jtype_i, n_neighbors,
                         dbdrs, nrb,
                         radial_coeffs, species_count, radial_funcs_count,
                         drf_drs);

        double *moment_values = (double *)calloc(alpha_moments_count, sizeof(double));
        double *moment_jac_rs = (double *)calloc(alpha_moments_count * n_neighbors * 3, sizeof(double));

        calc_basic_moments(
            n_neighbors,
            r_abs,
            r_unit,
            radial_funcs,
            drf_drs,
            alpha_index_basic,
            n_basic,
            moment_values,
            moment_jac_rs);

        double *moment_jac_cs = (double *)calloc(n_basic * species_count * radial_funcs_count * nrb, sizeof(double));
        double *moment_jac_rc = (double *)calloc(n_basic * species_count * radial_funcs_count * nrb * n_neighbors * 3, sizeof(double));

        calc_basic_moments_jac_radial_coeffs(
            n_neighbors, r_abs, r_unit, basis, dbdrs,
            alpha_index_basic, n_basic,
            species_count, jtype_i,
            radial_funcs_count, nrb,
            moment_jac_cs, moment_jac_rc);

        contract_moments_forward(moment_values, alpha_index_times, n_times);

        contract_moment_jacobians_forward(moment_values, moment_jac_rs, alpha_index_times, n_times, n_neighbors);

        double *dedmb = (double *)calloc(alpha_moments_count, sizeof(double));
        double *dgdmb = (double *)calloc(alpha_moments_count * n_neighbors * 3, sizeof(double));

        compute_dedmb_dgdmb(
            n_neighbors,
            alpha_index_times,
            n_times,
            alpha_moment_mapping,
            n_alpha_scalar,
            moment_coeffs,
            moment_values,
            moment_jac_rs,
            dedmb,
            dgdmb);

        accumulate_mbd_dedcs(
            itype,
            n_basic,
            species_count,
            radial_funcs_count,
            nrb,
            moment_jac_cs,
            dedmb,
            mbd_dedcs);

        accumulate_mbd_dgdcs_dsdcs(
            i,
            itype,
            n_atoms,
            n_neighbors,
            js_i,
            rs_i,
            n_basic,
            species_count,
            radial_funcs_count,
            nrb,
            moment_jac_cs,
            moment_jac_rc,
            dedmb,
            dgdmb,
            mbd_dgdcs,
            mbd_dsdcs);

        free(moment_jac_cs);
        free(moment_jac_rc);
        free(dedmb);
        free(dgdmb);

        accumulate_mbd_vatoms(
            i,
            n_atoms,
            n_alpha_scalar,
            alpha_moment_mapping,
            moment_values,
            mbd_vatoms);

        accumulate_mbd_dbdris_dbdeps(
            i,
            n_atoms,
            n_neighbors,
            js_i,
            rs_i,
            n_alpha_scalar,
            alpha_moment_mapping,
            moment_jac_rs,
            mbd_dbdris,
            mbd_dbdeps);

        for (int iamc = 0; iamc < n_alpha_scalar; iamc++)
        {
            int alpha_idx = alpha_moment_mapping[iamc];
            energies[i] += moment_coeffs[iamc] * moment_values[alpha_idx];
        }

        free(r_abs);
        free(r_unit);
        free(basis);
        free(ms_neighbors);
        free(radial_funcs);
        free(moment_values);
        free(moment_jac_rs);
    }
}

/* ==========================================================================
 * Training mode calculation with magnetic gradients
 * ========================================================================== */
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
    double *mbd_dsdcs)
{
    int rbs = radial_basis_size;
    int mbs = mag_basis_size;
    int nrb = rbs * mbs * mbs;

    for (int i = 0; i < n_atoms; i++)
    {
        energies[i] = species_coeffs[itypes[i]];
    }

    for (int i = 0; i < n_atoms; i++)
    {
        int itype = itypes[i];
        const double *rs_i = rs + i * n_neighbors * 3;
        const int *js_i = js + i * n_neighbors;
        const int *jtype_i = jtype + i * n_neighbors;

        double *r_abs = (double *)malloc(n_neighbors * sizeof(double));
        double *r_unit = (double *)malloc(n_neighbors * 3 * sizeof(double));

        calc_distances_and_unit_vectors(n_neighbors, rs_i, r_abs, r_unit);

        double *basis = (double *)malloc(4 * nrb * n_neighbors * sizeof(double));
        double *dbdrs = basis + nrb * n_neighbors;
        double *dbdmis = dbdrs + nrb * n_neighbors;
        double *dbdmjs = dbdmis + nrb * n_neighbors;

        double *ms_neighbors = (double *)malloc(n_neighbors * sizeof(double));
        for (int k = 0; k < n_neighbors; k++)
        {
            int j = js_i[k];
            ms_neighbors[k] = (j >= 0 && j < n_atoms) ? ms[j] : 0.0;
        }

        calc_radial_and_mag_basis(
            n_neighbors,
            r_abs,
            ms_neighbors,
            rbs,
            mbs,
            scaling,
            min_dist,
            max_dist,
            ms[i],
            min_mag,
            max_mag,
            basis,
            dbdrs,
            dbdmis,
            dbdmjs);

        store_mag_radial_basis_train(
            i, itype, n_atoms, n_neighbors, species_count, nrb,
            jtype_i, js_i,
            r_abs, r_unit, rs_i,
            basis, dbdrs,
            dbdmis, dbdmjs,
            rbd_values, rbd_dqdris, rbd_dqdmis, rbd_dqdeps);

        double *radial_funcs = (double *)malloc(4 * radial_funcs_count * n_neighbors * sizeof(double));
        double *drf_drs = radial_funcs + radial_funcs_count * n_neighbors;
        double *drf_dmis = drf_drs + radial_funcs_count * n_neighbors;
        double *drf_dmjs = drf_dmis + radial_funcs_count * n_neighbors;

        sum_radial_basis(itype, jtype_i, n_neighbors,
                         basis, nrb,
                         radial_coeffs, species_count, radial_funcs_count,
                         radial_funcs);

        sum_radial_basis(itype, jtype_i, n_neighbors,
                         dbdrs, nrb,
                         radial_coeffs, species_count, radial_funcs_count,
                         drf_drs);

        sum_radial_basis(itype, jtype_i, n_neighbors,
                         dbdmis, nrb,
                         radial_coeffs, species_count, radial_funcs_count,
                         drf_dmis);

        sum_radial_basis(itype, jtype_i, n_neighbors,
                         dbdmjs, nrb,
                         radial_coeffs, species_count, radial_funcs_count,
                         drf_dmjs);

        double *moment_values = (double *)calloc(alpha_moments_count, sizeof(double));
        double *moment_jac_rs = (double *)calloc(alpha_moments_count * n_neighbors * 3, sizeof(double));
        double *moment_jac_mis = (double *)calloc(alpha_moments_count * n_neighbors, sizeof(double));
        double *moment_jac_mjs = (double *)calloc(alpha_moments_count * n_neighbors, sizeof(double));

        calc_mag_basic_moments(
            n_neighbors,
            r_abs,
            r_unit,
            radial_funcs,
            drf_drs,
            drf_dmis,
            drf_dmjs,
            alpha_index_basic,
            n_basic,
            moment_values,
            moment_jac_rs,
            moment_jac_mis,
            moment_jac_mjs);

        double *moment_jac_cs = (double *)calloc(n_basic * species_count * radial_funcs_count * nrb, sizeof(double));
        double *moment_jac_rc = (double *)calloc(n_basic * species_count * radial_funcs_count * nrb * n_neighbors * 3, sizeof(double));
        double *moment_jac_mic = (double *)calloc(n_basic * species_count * radial_funcs_count * nrb * n_neighbors, sizeof(double));
        double *moment_jac_mjc = (double *)calloc(n_basic * species_count * radial_funcs_count * nrb * n_neighbors, sizeof(double));

        calc_mag_basic_moments_jac_radial_coeffs(
            n_neighbors, r_abs, r_unit, basis, dbdrs, dbdmis, dbdmjs,
            alpha_index_basic, n_basic,
            species_count, jtype_i,
            radial_funcs_count, nrb,
            moment_jac_cs, moment_jac_rc, moment_jac_mic, moment_jac_mjc);

        contract_moments_forward(moment_values, alpha_index_times, n_times);

        contract_moment_jacobians_forward(moment_values, moment_jac_rs, alpha_index_times, n_times, n_neighbors);

        contract_mag_moment_jacobians_forward(moment_values, moment_jac_mis, moment_jac_mjs,
                                              alpha_index_times, n_times, n_neighbors);

        double *dedmb = (double *)calloc(alpha_moments_count, sizeof(double));
        double *dgdmb = (double *)calloc(alpha_moments_count * n_neighbors * 3, sizeof(double));
        double *dgmidmb = (double *)calloc(alpha_moments_count * n_neighbors, sizeof(double));
        double *dgmjdmb = (double *)calloc(alpha_moments_count * n_neighbors, sizeof(double));

        compute_dedmb_dgdmb_dgmdmb(
            n_neighbors,
            alpha_index_times,
            n_times,
            alpha_moment_mapping,
            n_alpha_scalar,
            moment_coeffs,
            moment_values,
            moment_jac_rs,
            moment_jac_mis,
            moment_jac_mjs,
            dedmb,
            dgdmb,
            dgmidmb,
            dgmjdmb);

        accumulate_mbd_dedcs(
            itype,
            n_basic,
            species_count,
            radial_funcs_count,
            nrb,
            moment_jac_cs,
            dedmb,
            mbd_dedcs);

        accumulate_mbd_dgdcs_dsdcs(
            i,
            itype,
            n_atoms,
            n_neighbors,
            js_i,
            rs_i,
            n_basic,
            species_count,
            radial_funcs_count,
            nrb,
            moment_jac_cs,
            moment_jac_rc,
            dedmb,
            dgdmb,
            mbd_dgdcs,
            mbd_dsdcs);

        accumulate_mbd_dgmdcs(
            i,
            itype,
            n_atoms,
            n_neighbors,
            js_i,
            n_basic,
            species_count,
            radial_funcs_count,
            nrb,
            moment_jac_cs,
            moment_jac_mic,
            moment_jac_mjc,
            dedmb,
            dgmidmb,
            dgmjdmb,
            mbd_dgmdcs);

        free(moment_jac_cs);
        free(moment_jac_rc);
        free(moment_jac_mic);
        free(moment_jac_mjc);
        free(dedmb);
        free(dgdmb);
        free(dgmidmb);
        free(dgmjdmb);

        accumulate_mbd_vatoms(
            i,
            n_atoms,
            n_alpha_scalar,
            alpha_moment_mapping,
            moment_values,
            mbd_vatoms);

        accumulate_mbd_dbdris_dbdeps(
            i,
            n_atoms,
            n_neighbors,
            js_i,
            rs_i,
            n_alpha_scalar,
            alpha_moment_mapping,
            moment_jac_rs,
            mbd_dbdris,
            mbd_dbdeps);

        accumulate_mbd_dbdmis(
            i,
            n_atoms,
            n_neighbors,
            js_i,
            n_alpha_scalar,
            alpha_moment_mapping,
            moment_jac_mis,
            moment_jac_mjs,
            mbd_dbdmis);

        for (int iamc = 0; iamc < n_alpha_scalar; iamc++)
        {
            int alpha_idx = alpha_moment_mapping[iamc];
            energies[i] += moment_coeffs[iamc] * moment_values[alpha_idx];
        }

        free(r_abs);
        free(r_unit);
        free(basis);
        free(ms_neighbors);
        free(radial_funcs);
        free(moment_values);
        free(moment_jac_rs);
        free(moment_jac_mis);
        free(moment_jac_mjs);
    }
}
