import numba as nb
import numpy as np
import numpy.typing as npt


@nb.njit(
    nb.float64[:, :](
        nb.float64[:, :, :],
        nb.int64[:, :],
        nb.int64,
    ),
)
def _nb_forces_from_gradient(
    gradient: np.ndarray,
    all_js: np.ndarray,
    max_number_of_js: int,
) -> np.ndarray:
    number_of_atoms = gradient.shape[0]
    forces = np.zeros((number_of_atoms, 3))
    for i in range(number_of_atoms):
        for i_j in range(max_number_of_js):
            j = all_js[i, i_j]
            for k in range(3):
                forces[i, k] += gradient[i, i_j, k]
                forces[j, k] -= gradient[i, i_j, k]
    return forces


@nb.njit
def _calc_r_unit_pows(r_unit: np.ndarray, max_pow: int) -> np.ndarray:
    number_of_js = r_unit.shape[0]
    r_unit_pows = np.ones((max_pow + 1, number_of_js, 3))
    for pow in range(1, max_pow + 1):
        for j in range(number_of_js):
            for k in range(3):
                r_unit_pows[pow, j, k] = r_unit_pows[pow - 1, j, k] * r_unit[j, k]
    return r_unit_pows


@nb.njit(
    (
        nb.float64[:],
        nb.float64[:, :],
        nb.int64[:, :],
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64[:],
        nb.float64[:, :, :],
    ),
)
def _calc_moment_basic(
    r_abs,
    r_ijs_unit,
    alpha_index_basic,
    rb_values,
    rb_derivs,
    moment_components,
    moment_jacobian,
) -> None:
    """Compute basic moment components and its jacobian wrt `r_ijs`."""
    # Precompute powers
    max_pow = int(np.max(alpha_index_basic))
    r_unit_pows = _calc_r_unit_pows(r_ijs_unit, max_pow)

    number_of_js = moment_jacobian.shape[1]
    for aib_i, aib in enumerate(alpha_index_basic):
        mu, xpow, ypow, zpow = aib
        xyzpow = xpow + ypow + zpow
        for j in range(number_of_js):
            mult0 = 1.0
            mult0 *= r_unit_pows[xpow, j, 0]
            mult0 *= r_unit_pows[ypow, j, 1]
            mult0 *= r_unit_pows[zpow, j, 2]
            moment_components[aib_i] += rb_values[mu, j] * mult0
            for k in range(3):
                moment_jacobian[aib_i, j, k] = (
                    r_ijs_unit[j, k]
                    * mult0
                    * (rb_derivs[mu, j] - xyzpow * rb_values[mu, j] / r_abs[j])
                )
            if xpow != 0:
                moment_jacobian[aib_i, j, 0] += (
                    rb_values[mu, j]
                    * (xpow * r_unit_pows[xpow - 1, j, 0])
                    * r_unit_pows[ypow, j, 1]
                    * r_unit_pows[zpow, j, 2]
                    / r_abs[j]
                )
            if ypow != 0:
                moment_jacobian[aib_i, j, 1] += (
                    rb_values[mu, j]
                    * r_unit_pows[xpow, j, 0]
                    * (ypow * r_unit_pows[ypow - 1, j, 1])
                    * r_unit_pows[zpow, j, 2]
                    / r_abs[j]
                )
            if zpow != 0:
                moment_jacobian[aib_i, j, 2] += (
                    rb_values[mu, j]
                    * r_unit_pows[xpow, j, 0]
                    * r_unit_pows[ypow, j, 1]
                    * (zpow * r_unit_pows[zpow - 1, j, 2])
                    / r_abs[j]
                )


@nb.njit((nb.int64[:, :], nb.float64[:], nb.float64[:, :, :]))
def _contract_moments(
    alpha_index_times: npt.NDArray[np.int64],
    moment_values: npt.NDArray[np.float64],
    moment_jac_rs: npt.NDArray[np.float64],
) -> None:
    """Compute contractions of moments."""
    number_of_js = moment_jac_rs.shape[1]
    for ait in alpha_index_times:
        i1, i2, mult, i3 = ait
        moment_values[i3] += mult * moment_values[i1] * moment_values[i2]
        for j in range(number_of_js):
            for k in range(3):
                moment_jac_rs[i3, j, k] += mult * (
                    moment_jac_rs[i1, j, k] * moment_values[i2]
                    + moment_values[i1] * moment_jac_rs[i2, j, k]
                )


# @nb.njit(
#     nb.float64[:, :](
#         nb.int64[:, :],
#         nb.int64[:],
#         nb.float64[:],
#         nb.float64[:],
#         nb.float64[:, :, :],
#     ),
# )
# def _propagate_forward(
#     alpha_index_times,
#     alpha_moment_mapping,
#     moment_coeffs,
#     moment_components,
#     moment_jacobian,
# ):
#     """Calculate gradients using the forward propagation."""
#     _calc_moment_times(alpha_index_times, moment_components, moment_jacobian)
#     _, number_of_js, _ = moment_jacobian.shape
#     gradient = np.zeros((number_of_js, 3))
#     for basis_i, moment_i in enumerate(alpha_moment_mapping):
#         for j in range(number_of_js):
#             for k in range(3):
#                 gradient[j, k] += (
#                     moment_coeffs[basis_i] * moment_jacobian[moment_i, j, k]
#                 )

#     return gradient


@nb.njit(
    nb.float64[:, :](
        nb.int64[:, :],
        nb.int64[:, :],
        nb.int64[:],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:, :, :],
    ),
)
def _propagate_backward(
    alpha_index_basic,
    alpha_index_times,
    alpha_moment_mapping,
    moment_coeffs,
    moment_components,
    moment_jacobian,
):
    # alternatively with backpropagation: (saves in the order of 20% for higher levels)
    alpha_moments_count, number_of_js, _ = moment_jacobian.shape
    tmp_moment_ders = np.zeros((alpha_moments_count))
    for basis_i, moment_i in enumerate(alpha_moment_mapping):
        tmp_moment_ders[moment_i] = moment_coeffs[basis_i]
    for ait in alpha_index_times[::-1]:
        i1, i2, mult, i3 = ait
        tmp_moment_ders[i2] += tmp_moment_ders[i3] * mult * moment_components[i1]
        tmp_moment_ders[i1] += tmp_moment_ders[i3] * mult * moment_components[i2]

    gradient = np.zeros((number_of_js, 3))
    for aib_i in range(alpha_index_basic.shape[0]):
        for j in range(number_of_js):
            for k in range(3):
                gradient[j, k] += tmp_moment_ders[aib_i] * moment_jacobian[aib_i, j, k]

    return gradient


@nb.njit(
    nb.types.Tuple((nb.float64, nb.float64[:, :]))(
        nb.float64[:, :],
        nb.float64[:],
        nb.float64[:, :],
        nb.float64[:, :],
        nb.int64,
        nb.int64[:],
        nb.int64[:, :],
        nb.int64[:, :],
        nb.float64[:],
    )
)
def _nb_calc_local_energy_and_gradient(
    r_ijs_unit,
    r_abs,
    rb_values,
    rb_derivs,
    alpha_moments_count,
    alpha_moment_mapping,
    alpha_index_basic,
    alpha_index_times,
    moment_coeffs,
):
    (number_of_js,) = r_abs.shape
    moment_components = np.zeros(alpha_moments_count)
    moment_jacobian = np.zeros((alpha_moments_count, number_of_js, 3))

    _calc_moment_basic(
        r_abs,
        r_ijs_unit,
        alpha_index_basic,
        rb_values,
        rb_derivs,
        moment_components,
        moment_jacobian,
    )

    # For moments and energy:
    # Compute moment contraction components
    for ait in alpha_index_times:
        i1, i2, mult, i3 = ait
        moment_components[i3] += mult * moment_components[i1] * moment_components[i2]
    # Extract basis elements and multiply with moment coefficients
    energy = 0.0
    for basis_i, moment_i in enumerate(alpha_moment_mapping):
        energy += moment_coeffs[basis_i] * moment_components[moment_i]

    gradient = _propagate_backward(
        alpha_index_basic,
        alpha_index_times,
        alpha_moment_mapping,
        moment_coeffs,
        moment_components,
        moment_jacobian,
    )

    return energy, gradient


@nb.njit(
    (
        nb.int64,
        nb.int64[:],
        nb.float64[:],
        nb.float64[:, :],
        nb.int64[:, :],
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64[:, :, :, :],
        nb.float64[:],
        nb.float64[:, :, :],
        nb.float64[:, :, :, :],
        nb.float64[:, :, :, :, :, :],
    ),
)
def _calc_moment_basic_with_jacobian_radial_coeffs(
    itype,
    jtypes,
    r_abs: np.ndarray,
    r_ijs_unit: np.ndarray,
    alpha_index_basic: np.ndarray,
    rb_values: np.ndarray,
    rb_derivs: np.ndarray,
    rb_coeffs: np.ndarray,
    moment_components: np.ndarray,
    moment_jacobian: np.ndarray,
    moment_jac_cs: np.ndarray,
    moment_jac_rc: np.ndarray,
) -> None:
    """Compute basic moment components and its jacobian wrt `r_ijs`."""
    # Precompute powers
    max_pow = int(np.max(alpha_index_basic))
    r_unit_pows = _calc_r_unit_pows(r_ijs_unit, max_pow)

    rbs = rb_coeffs.shape[3]
    der = np.zeros(3)
    for aib_i, aib in enumerate(alpha_index_basic):
        mu, xpow, ypow, zpow = aib
        xyzpow = xpow + ypow + zpow
        for j, jtype in enumerate(jtypes):
            mult0 = 1.0
            mult0 *= r_unit_pows[xpow, j, 0]
            mult0 *= r_unit_pows[ypow, j, 1]
            mult0 *= r_unit_pows[zpow, j, 2]
            for ib in range(rbs):
                val = rb_values[ib, j] * mult0
                for k in range(3):
                    der[k] = (
                        r_ijs_unit[j, k]
                        * mult0
                        * (rb_derivs[ib, j] - xyzpow * rb_values[ib, j] / r_abs[j])
                    )
                if xpow != 0:
                    der[0] += (
                        rb_values[ib, j]
                        * (xpow * r_unit_pows[xpow - 1, j, 0])
                        * r_unit_pows[ypow, j, 1]
                        * r_unit_pows[zpow, j, 2]
                        / r_abs[j]
                    )
                if ypow != 0:
                    der[1] += (
                        rb_values[ib, j]
                        * r_unit_pows[xpow, j, 0]
                        * (ypow * r_unit_pows[ypow - 1, j, 1])
                        * r_unit_pows[zpow, j, 2]
                        / r_abs[j]
                    )
                if zpow != 0:
                    der[2] += (
                        rb_values[ib, j]
                        * r_unit_pows[xpow, j, 0]
                        * r_unit_pows[ypow, j, 1]
                        * (zpow * r_unit_pows[zpow - 1, j, 2])
                        / r_abs[j]
                    )
                c = rb_coeffs[itype, jtype, mu, ib]

                moment_components[aib_i] += c * val
                for k in range(3):
                    moment_jacobian[aib_i, j, k] += c * der[k]
                moment_jac_cs[aib_i, jtype, mu, ib] += val
                for k in range(3):
                    moment_jac_rc[aib_i, jtype, mu, ib, j, k] += der[k]


@nb.njit(nb.float64[:, :, :](nb.int64[:], nb.float64[:], nb.float64[:, :, :, :]))
def _calc_dedcs(
    alpha_moment_mapping: np.ndarray,
    moment_coeffs: np.ndarray,
    moment_jac_cs: np.ndarray,
) -> None:
    spc, rfc, rbs = moment_jac_cs.shape[1:]
    dedcs = np.zeros((spc, rfc, rbs))
    for i, j in enumerate(alpha_moment_mapping):
        for ispc in range(spc):
            for irfc in range(rfc):
                for irbs in range(rbs):
                    dedcs[ispc, irfc, irbs] += (
                        moment_jac_cs[j, ispc, irfc, irbs] * moment_coeffs[i]
                    )
    return dedcs


@nb.njit(
    nb.types.Tuple((nb.float64[:, :, :], nb.float64[:, :, :, :, :]))(
        nb.int64,
        nb.int64[:, :],
        nb.int64[:],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:, :, :],
        nb.float64[:, :, :, :],
        nb.float64[:, :, :, :, :, :],
    ),
)
def cald_dedcs_and_dgdcs(
    alpha_index_basic_count: np.int64,
    alpha_index_times: np.ndarray,
    alpha_moment_mapping: np.ndarray,
    moment_coeffs: np.ndarray,
    moment_values: np.ndarray,
    moment_jac_rs: np.ndarray,
    moment_jac_cs: np.ndarray,
    moment_jac_rc: np.ndarray,
) -> np.ndarray:
    """Calculate dV/dc and d(dV/dr)/dc.

    - dV/dc: Jacobian of site energy to radial basis coefficients
    - d(dV/dr)/dc: Jacobians of site energy gradients to radial basis coefficients

    Returns
    -------
    dedcs : np.ndarray
        dV/dc.
    dgdcs : np.ndarray
        d(dV/dr)/dc.

    """
    _, spc, rfc, rbs, nns, _ = moment_jac_rc.shape

    dedmb = np.zeros_like(moment_values)
    dgdmb = np.zeros_like(moment_jac_rs)
    dedmb[alpha_moment_mapping] = moment_coeffs  # dV/dB
    for ait in alpha_index_times[::-1]:
        i1, i2, mult, i3 = ait
        dedmb[i1] += mult * dedmb[i3] * moment_values[i2]
        dedmb[i2] += mult * dedmb[i3] * moment_values[i1]
    for ait in alpha_index_times:
        i1, i2, mult, i3 = ait
        for j in range(nns):
            for ixyz0 in range(3):
                dgdmb[i1, j, ixyz0] += mult * dedmb[i3] * moment_jac_rs[i2, j, ixyz0]
                dgdmb[i2, j, ixyz0] += mult * dedmb[i3] * moment_jac_rs[i1, j, ixyz0]
    for ait in alpha_index_times[::-1]:
        i1, i2, mult, i3 = ait
        for j in range(nns):
            for ixyz0 in range(3):
                dgdmb[i1, j, ixyz0] += mult * dgdmb[i3, j, ixyz0] * moment_values[i2]
                dgdmb[i2, j, ixyz0] += mult * dgdmb[i3, j, ixyz0] * moment_values[i1]

    dedcs = np.zeros((spc, rfc, rbs))
    for iamc in range(alpha_index_basic_count):
        v1 = dedmb[iamc]
        for ispc in range(spc):
            for irfc in range(rfc):
                for irbs in range(rbs):
                    v0 = moment_jac_cs[iamc, ispc, irfc, irbs]
                    dedcs[ispc, irfc, irbs] += v0 * v1

    dgdcs = np.zeros(moment_jac_rc.shape[1:])
    for iamc in range(alpha_index_basic_count):
        v1 = dedmb[iamc]
        for ispc in range(spc):
            for irfc in range(rfc):
                for irbs in range(rbs):
                    for j in range(nns):
                        for ixyz in range(3):
                            v0 = moment_jac_rc[iamc, ispc, irfc, irbs, j, ixyz]
                            dgdcs[ispc, irfc, irbs, j, ixyz] += v0 * v1
    for iamc in range(alpha_index_basic_count):
        for ispc in range(spc):
            for irfc in range(rfc):
                for irbs in range(rbs):
                    v0 = moment_jac_cs[iamc, ispc, irfc, irbs]
                    for j in range(nns):
                        for ixyz in range(3):
                            v1 = dgdmb[iamc, j, ixyz]
                            dgdcs[ispc, irfc, irbs, j, ixyz] += v0 * v1

    return dedcs, dgdcs


@nb.njit(
    nb.types.Tuple(
        (
            nb.float64[:],
            nb.float64[:, :, :],
            nb.float64[:, :, :],
            nb.float64[:, :, :, :, :],
        ),
    )(
        nb.int64,
        nb.int64[:],
        nb.float64[:],
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64[:, :, :, :],
        nb.int64,
        nb.int64[:],
        nb.int64[:, :],
        nb.int64[:, :],
        nb.float64[:],
    ),
)
def _nb_calc_moment(
    itype,
    jtypes,
    r_abs: np.ndarray,
    r_ijs_unit: np.ndarray,
    rb_values: np.ndarray,
    rb_derivs: np.ndarray,
    rb_coeffs: np.ndarray,
    alpha_moments_count: np.int64,
    alpha_moment_mapping: np.ndarray,
    alpha_index_basic: np.ndarray,
    alpha_index_times: np.ndarray,
    moment_coeffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, species_count, rfs, rbs = rb_coeffs.shape
    amc = alpha_moments_count
    moment_values = np.zeros(amc)
    moment_jac_rs = np.zeros((amc, *r_ijs_unit.shape))
    moment_jac_cs = np.zeros((amc, species_count, rfs, rbs))
    moment_jac_rc = np.zeros((amc, species_count, rfs, rbs, *r_ijs_unit.shape))

    _calc_moment_basic_with_jacobian_radial_coeffs(
        itype,
        jtypes,
        r_abs,
        r_ijs_unit,
        alpha_index_basic,
        rb_values,
        rb_derivs,
        rb_coeffs,
        moment_values,
        moment_jac_rs,
        moment_jac_cs,
        moment_jac_rc,
    )

    _contract_moments(alpha_index_times, moment_values, moment_jac_rs)

    dedcs, dgdcs = cald_dedcs_and_dgdcs(
        alpha_index_basic.shape[0],
        alpha_index_times,
        alpha_moment_mapping,
        moment_coeffs,
        moment_values,
        moment_jac_rs,
        moment_jac_cs,
        moment_jac_rc,
    )

    return (
        moment_values[alpha_moment_mapping],
        moment_jac_rs[alpha_moment_mapping],
        dedcs,
        dgdcs,
    )


@nb.njit
def _store_radial_basis_values(
    i: np.int64,
    itype: np.int64,
    js: np.ndarray,
    jtypes: np.ndarray,
    r_ijs: np.ndarray,
    r_ijs_unit: np.ndarray,
    basis_vs: np.ndarray,
    basis_ds: np.ndarray,
    values: np.ndarray,
    dqdris: np.ndarray,
    dqdeps: np.ndarray,
) -> None:
    radial_basis_size = basis_vs.shape[0]
    for ib in range(radial_basis_size):
        for k, j in enumerate(js):
            jtype = jtypes[k]
            values[itype, jtype, ib] += basis_vs[ib, k]
            for ixyz0 in range(3):
                tmp = basis_ds[ib, k] * r_ijs_unit[k, ixyz0]
                dqdris[itype, jtype, ib, i, ixyz0] -= tmp
                dqdris[itype, jtype, ib, j, ixyz0] += tmp
                for ixyz1 in range(3):
                    dqdeps[itype, jtype, ib, ixyz0, ixyz1] += tmp * r_ijs[k, ixyz1]
