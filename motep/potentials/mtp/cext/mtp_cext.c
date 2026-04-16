#include "mtp_cext.h"
#include "mtp_cext_kernels.h"

/* ============================================================================
 * Main run mode calculation
 * ============================================================================ */
void calc_run(
    int n_atoms, int n_neighbors,
    const double *rs,
    const int *itypes, const int *jtype,
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
    double *energies, double *gradient,
    double *mbd_vatoms)
{
    int rbs = radial_basis_size;

    /* Initialize energies with species baseline */
    for (int i = 0; i < n_atoms; i++)
    {
        energies[i] = species_coeffs[itypes[i]];
    }

    /* Initialize gradient to zero */
    memset(gradient, 0, n_atoms * n_neighbors * 3 * sizeof(double));

    /* ========================================================================
     * Main loop over atoms
     * ======================================================================== */
    for (int i = 0; i < n_atoms; i++)
    {
        int itype = itypes[i];
        const double *rs_i = rs + i * n_neighbors * 3;
        const int *jtype_i = jtype + i * n_neighbors;
        double *grad_i = gradient + i * n_neighbors * 3;

        /* ====================================================================
         * Step 1: Calculate distances and unit vectors
         * ==================================================================== */
        double *r_abs = (double *)malloc(n_neighbors * sizeof(double));
        double *r_unit = (double *)malloc(n_neighbors * 3 * sizeof(double));

        calc_distances_and_unit_vectors(n_neighbors, rs_i, r_abs, r_unit);

        /* ====================================================================
         * Step 2: Calculate radial basis for each neighbor
         * Layout: [radial_basis_size][n_neighbors] for cache-efficient access
         * ==================================================================== */
        double *radial_basis = (double *)malloc(2 * rbs * n_neighbors * sizeof(double));
        double *drb_drs = radial_basis + rbs * n_neighbors;

        /* Reusable temporary buffers for chebyshev_basis output */
        double *tmp_rb_vals = (double *)malloc(rbs * sizeof(double));
        double *tmp_drb_drs = (double *)malloc(rbs * sizeof(double));

        for (int j = 0; j < n_neighbors; j++)
        {
            chebyshev_basis(r_abs[j], rbs, scaling, min_dist, max_dist,
                            tmp_rb_vals, tmp_drb_drs);

            /* Transpose to basis-major layout: [rbs][n_neighbors] */
            for (int irb = 0; irb < rbs; irb++)
            {
                radial_basis[irb * n_neighbors + j] = tmp_rb_vals[irb];
                drb_drs[irb * n_neighbors + j] = tmp_drb_drs[irb];
            }
        }

        free(tmp_rb_vals);
        free(tmp_drb_drs);

        /* ====================================================================
         * Step 3: Sum radial basis functions with coefficients
         * ==================================================================== */
        double *radial_funcs = (double *)malloc(2 * radial_funcs_count * n_neighbors * sizeof(double));
        double *drf_drs = radial_funcs + radial_funcs_count * n_neighbors;

        /* Sum values: radial_basis[0:rbs] contains basis values */
        sum_radial_basis(itype, jtype_i, n_neighbors,
                         radial_basis, rbs,
                         radial_coeffs, species_count, radial_funcs_count,
                         radial_funcs);

        /* Sum derivatives: rb_vals[rbs:2*rbs] contains basis derivatives */
        sum_radial_basis(itype, jtype_i, n_neighbors,
                         drb_drs, rbs,
                         radial_coeffs, species_count, radial_funcs_count,
                         drf_drs);

        /* ====================================================================
         * Step 4: Calculate basic moment values and jacobians
         * ==================================================================== */
        double *m = (double *)calloc(alpha_moments_count, sizeof(double));
        double *m_jac = (double *)calloc(n_basic * n_neighbors * 3, sizeof(double));

        calc_basic_moments(n_neighbors, r_abs, r_unit, radial_funcs, drf_drs,
                           alpha_index_basic, n_basic,
                           m, m_jac);

        /* ====================================================================
         * Step 5: Contract moments (forward), compute gradient (backward)
         * ==================================================================== */
        contract_moments_forward(m, alpha_index_times, n_times);

        contract_moment_jacobians_backwards(
            n_basic, alpha_moments_count,
            n_alpha_scalar,
            alpha_moment_mapping, alpha_index_times, n_times,
            moment_coeffs, m, m_jac, n_neighbors, grad_i);

        for (int i_am = 0; i_am < n_alpha_scalar; i_am++)
        {
            int idx = alpha_moment_mapping[i_am];
            energies[i] += moment_coeffs[i_am] * m[idx];
        }

        accumulate_mbd_vatoms(
            i,
            n_atoms,
            n_alpha_scalar,
            alpha_moment_mapping,
            m,
            mbd_vatoms);

        free(r_abs);
        free(r_unit);
        free(radial_basis);
        free(radial_funcs);
        free(m_jac);
        free(m);
    }
}

/* ============================================================================
 * Calculate forces from gradients
 * ============================================================================ */
void calc_forces_from_gradient(
    const double *gradient,
    const int *js,
    int n_atoms, int n_neighbors,
    double *forces)
{
    /* Initialize forces to zero */
    memset(forces, 0, n_atoms * 3 * sizeof(double));

    /* Accumulate forces from gradients */
    for (int i = 0; i < n_atoms; i++)
    {
        for (int k = 0; k < n_neighbors; k++)
        {
            int j = js[i * n_neighbors + k];
            if (j < 0)
                continue; /* Skip padding entries */
            for (int xyz = 0; xyz < 3; xyz++)
            {
                double grad_val = gradient[(i * n_neighbors + k) * 3 + xyz];
                forces[i * 3 + xyz] += grad_val;
                forces[j * 3 + xyz] -= grad_val;
            }
        }
    }
}

/* ============================================================================
 * Training mode calculation
 * ============================================================================ */
void calc_train(
    int n_atoms, int n_neighbors,
    const double *rs,
    const int *js,
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
    MomentBasisData *mbd)
{
    int rbs = radial_basis_size;

    /* Note: rbd/mbd arrays are normally zeroed before enering this function */

    /* ========================================================================
     * Main loop over atoms - compute training mode
     * ======================================================================== */
    for (int i = 0; i < n_atoms; i++)
    {
        int itype = itypes[i];
        const double *rs_i = rs + i * n_neighbors * 3;
        const int *jtype_i = jtype + i * n_neighbors;

        /* ====================================================================
         * Step 1: Calculate distances and unit vectors
         * ==================================================================== */
        double *r_abs = (double *)malloc(n_neighbors * sizeof(double));
        double *r_unit = (double *)malloc(n_neighbors * 3 * sizeof(double));

        calc_distances_and_unit_vectors(n_neighbors, rs_i, r_abs, r_unit);

        /* ====================================================================
         * Step 2: Calculate radial basis and derivatives
         * Layout: [radial_basis_size][n_neighbors]
         * ==================================================================== */
        double *radial_basis = (double *)malloc(2 * rbs * n_neighbors * sizeof(double));
        double *drb_drs = radial_basis + rbs * n_neighbors;

        double *tmp_rb_vals = (double *)malloc(rbs * sizeof(double));
        double *tmp_drb_drs = (double *)malloc(rbs * sizeof(double));

        for (int j = 0; j < n_neighbors; j++)
        {
            chebyshev_basis(r_abs[j], rbs, scaling, min_dist, max_dist,
                            tmp_rb_vals, tmp_drb_drs);

            /* Transpose to basis-major layout: [rbs][n_neighbors] */
            for (int irb = 0; irb < rbs; irb++)
            {
                radial_basis[irb * n_neighbors + j] = tmp_rb_vals[irb];
                drb_drs[irb * n_neighbors + j] = tmp_drb_drs[irb];
            }
        }

        free(tmp_rb_vals);
        free(tmp_drb_drs);
        /* ====================================================================
         * Step 3: Store radial basis values and derivatives
         * ==================================================================== */
        const int *js_i = js + i * n_neighbors;
        store_radial_basis_train(
            i, itype, n_atoms, n_neighbors, species_count, rbs,
            jtype_i, js_i,
            r_abs, r_unit, rs_i,
            radial_basis, drb_drs,
            rbd->values, rbd->dqdris, rbd->dqdeps);

        /* ====================================================================
         * Step 4: Sum radial basis functions with coefficients
         * ==================================================================== */
        double *radial_funcs = (double *)malloc(2 * radial_funcs_count * n_neighbors * sizeof(double));
        double *drf_drs = radial_funcs + radial_funcs_count * n_neighbors;

        sum_radial_basis(itype, jtype_i, n_neighbors,
                         radial_basis, rbs,
                         radial_coeffs, species_count, radial_funcs_count,
                         radial_funcs);

        sum_radial_basis(itype, jtype_i, n_neighbors,
                         drb_drs, rbs,
                         radial_coeffs, species_count, radial_funcs_count,
                         drf_drs);

        /* ====================================================================
         * Step 5: Allocate moment arrays and calculate basic moments
         * ==================================================================== */
        double *m = (double *)calloc(alpha_moments_count, sizeof(double));
        double *m_jac = (double *)calloc(alpha_moments_count * n_neighbors * 3, sizeof(double));

        calc_basic_moments(n_neighbors, r_abs, r_unit, radial_funcs, drf_drs,
                           alpha_index_basic, n_basic,
                           m, m_jac);

        /* ====================================================================
         * Step 5b: Compute jacobians of basic moments w.r.t. radial coefficients
         * ==================================================================== */
        double *moment_jac_cs = (double *)calloc(n_basic * species_count * radial_funcs_count * rbs, sizeof(double));
        double *moment_jac_rc = (double *)calloc(n_basic * species_count * radial_funcs_count * rbs * n_neighbors * 3, sizeof(double));

        calc_basic_moments_jac_radial_coeffs(
            n_neighbors, r_abs, r_unit, radial_basis, drb_drs,
            alpha_index_basic, n_basic,
            species_count, jtype_i,
            radial_funcs_count, rbs,
            moment_jac_cs, moment_jac_rc);

        /* ====================================================================
         * Step 6: Contract moments through alpha_index_times
         * ==================================================================== */
        /* Contract moment values */
        contract_moments_forward(m, alpha_index_times, n_times);

        /* Contract moment jacobians through alpha_index_times indices */
        contract_moment_jacobians_forward(m, m_jac, alpha_index_times, n_times, n_neighbors);

        /* ====================================================================
         * Step 6a: Compute dedmb and dgdmb (backprop through moment contractions)
         * ==================================================================== */
        double *dedmb = (double *)calloc(alpha_moments_count, sizeof(double));
        double *dgdmb = (double *)calloc(alpha_moments_count * n_neighbors * 3, sizeof(double));

        compute_dedmb_dgdmb(
            n_neighbors,
            alpha_index_times,
            n_times,
            alpha_moment_mapping,
            n_alpha_scalar,
            moment_coeffs,
            m,
            m_jac,
            dedmb,
            dgdmb);

        /* ====================================================================
         * Step 6b: Accumulate dvdcs from basic moment jacobians
         * ==================================================================== */
        accumulate_mbd_dvdcs(
            i,
            itype,
            n_atoms,
            n_basic,
            species_count,
            radial_funcs_count,
            rbs,
            moment_jac_cs,
            dedmb,
            mbd->dvdcs);

        /* ====================================================================
         * Step 6c: Accumulate dgdcs from basic moment jacobians
         * ==================================================================== */
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
            rbs,
            moment_jac_cs,
            moment_jac_rc,
            dedmb,
            dgdmb,
            mbd->dgdcs,
            mbd->dsdcs);

        free(moment_jac_cs);
        free(moment_jac_rc);
        free(dedmb);
        free(dgdmb);

        /* ====================================================================
         * Step 7: Accumulate mapped moments into mbd_vatoms
         * ==================================================================== */
        accumulate_mbd_vatoms(
            i,
            n_atoms,
            n_alpha_scalar,
            alpha_moment_mapping,
            m,
            mbd->vatoms);

        /* ====================================================================
         * Step 8: Accumulate moment basis derivatives
         * ==================================================================== */
        accumulate_mbd_dbdris_dbdeps(
            i,
            n_atoms,
            n_neighbors,
            js_i,
            rs_i,
            n_alpha_scalar,
            alpha_moment_mapping,
            m_jac,
            mbd->dbdris,
            mbd->dbdeps);

        /* ====================================================================
         * Step 9: Compute energy from moment coefficients for this atom
         * ==================================================================== */
        /* Initialize energies with species baseline */
        /* Use the Neumaier summation algorithm to reduce floating-point error */
        double sum = species_coeffs[itype];
        double c = 0.0;
        for (int iamc = 0; iamc < n_alpha_scalar; iamc++)
        {
            int alpha_idx = alpha_moment_mapping[iamc];
            double x = moment_coeffs[iamc] * m[alpha_idx];
            double t = sum + x;
            if (fabs(sum) >= fabs(x))
                c += (sum - t) + x;
            else
                c += (x - t) + sum;
            sum = t;
        }
        energies[i] = sum + c;

        /* ====================================================================
         * Cleanup
         * ==================================================================== */
        free(r_abs);
        free(r_unit);
        free(radial_basis);
        free(radial_funcs);
        free(m_jac);
        free(m);
    }
}
