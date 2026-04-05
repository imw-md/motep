import numba as nb
import numpy as np
import numpy.typing as npt

from motep.potentials.mtp.numba.moment import _calc_r_unit_pows


@nb.njit(
    (
        nb.int32,
        nb.int32[:],
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64[:, :],
    ),
)
def update_mbd_dbdmis(
    i: np.int32,
    js: npt.NDArray[np.int32],
    mbd_dbdmis: npt.NDArray[np.float64],
    basis_jac_mis: npt.NDArray[np.float64],
    basis_jac_mjs: npt.NDArray[np.float64],
) -> None:
    for iamc in range(mbd_dbdmis.shape[0]):
        for i_j, j in enumerate(js):
            mbd_dbdmis[iamc, i] += basis_jac_mis[iamc, i_j]
            mbd_dbdmis[iamc, j] += basis_jac_mjs[iamc, i_j]


@nb.njit(
    (
        nb.int32,
        nb.int32,
        nb.int32[:],
        nb.float64[:, :, :, :, :],
        nb.float64[:, :, :, :],
        nb.float64[:, :, :, :],
    ),
)
def update_mbd_dgmdcs(
    i: np.int32,
    itype: np.int32,
    js: npt.NDArray[np.int32],
    mbd_dgmdcs: npt.NDArray[np.float64],
    tmp_dgmidcs: npt.NDArray[np.float64],
    tmp_dgmjdcs: npt.NDArray[np.float64],
) -> None:
    s1, s2, s3 = mbd_dgmdcs.shape[1:4]
    for i1 in range(s1):
        for i2 in range(s2):
            for i3 in range(s3):
                for k, j in enumerate(js):
                    vi = tmp_dgmidcs[i1, i2, i3, k]
                    vj = tmp_dgmjdcs[i1, i2, i3, k]
                    mbd_dgmdcs[itype, i1, i2, i3, i] += vi
                    mbd_dgmdcs[itype, i1, i2, i3, j] += vj


@nb.njit(
    (
        nb.int32[:, :],
        nb.float64[:],
        nb.float64[:, :, :],
        nb.float64[:, :],
        nb.float64[:, :],
    )
)
def _mag_contract_moments(
    alpha_index_times: npt.NDArray[np.int32],
    moment_values: npt.NDArray[np.float64],
    moment_jac_rs: npt.NDArray[np.float64],
    moment_jac_mis: npt.NDArray[np.float64],
    moment_jac_mjs: npt.NDArray[np.float64],
) -> None:
    """Compute moment contractions."""
    # Contract the moment values
    for ait in alpha_index_times:
        i1, i2, mult, i3 = ait
        moment_values[i3] += mult * moment_values[i1] * moment_values[i2]

    # Contract the jacobians of all moment components
    for ait in alpha_index_times:
        i1, i2, mult, i3 = ait
        for j in range(moment_jac_rs.shape[1]):
            moment_jac_mis[i3, j] += mult * (
                moment_jac_mis[i1, j] * moment_values[i2]
                + moment_values[i1] * moment_jac_mis[i2, j]
            )
            moment_jac_mjs[i3, j] += mult * (
                moment_jac_mjs[i1, j] * moment_values[i2]
                + moment_values[i1] * moment_jac_mjs[i2, j]
            )
            for k in range(3):
                moment_jac_rs[i3, j, k] += mult * (
                    moment_jac_rs[i1, j, k] * moment_values[i2]
                    + moment_values[i1] * moment_jac_rs[i2, j, k]
                )


@nb.njit(
    (
        nb.int32[:, :],
        nb.int32[:, :],
        nb.int32[:],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:, :, :],
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64[:],
        nb.float64[:],
    ),
)
def _mag_contract_moments_backwards(
    alpha_index_basic,
    alpha_index_times,
    alpha_moment_mapping,
    moment_coeffs,
    moment_values,
    moment_jac_rs,
    moment_jac_mis,
    moment_jac_mjs,
    grad,
    grad_mag_i,
    grad_mag_j,
):
    """Compute moment contractions.

    Propagates backwards for the gradient and returns it without computing all
    moment jacobians. This speeds up around 4 times for level 20.

    """
    # First do the values as usual.
    for i1, i2, mult, i3 in alpha_index_times:
        moment_values[i3] += mult * moment_values[i1] * moment_values[i2]

    amc, njs, _ = moment_jac_rs.shape
    # Now go backwards for the jacobians/gradient
    tmp_moment_ders = np.zeros(amc)
    tmp_moment_ders[alpha_moment_mapping] = moment_coeffs
    for ait in alpha_index_times[::-1]:
        i1, i2, mult, i3 = ait
        tmp_moment_ders[i2] += tmp_moment_ders[i3] * mult * moment_values[i1]
        tmp_moment_ders[i1] += tmp_moment_ders[i3] * mult * moment_values[i2]

    for aib_i in range(alpha_index_basic.shape[0]):
        for j in range(njs):
            grad_mag_i[j] += tmp_moment_ders[aib_i] * moment_jac_mis[aib_i, j]
            grad_mag_j[j] += tmp_moment_ders[aib_i] * moment_jac_mjs[aib_i, j]
            for k in range(3):
                grad[j, k] += tmp_moment_ders[aib_i] * moment_jac_rs[aib_i, j, k]


@nb.njit(
    (
        nb.int32,
        nb.int32[:],
        nb.float64[:],
        nb.float64[:, :],
        nb.int32[:, :],
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64[:, :, :, :],
        nb.float64[:],
        nb.float64[:, :, :],
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64[:, :, :, :],
        nb.float64[:, :, :, :, :, :],
        nb.float64[:, :, :, :, :],
        nb.float64[:, :, :, :, :],
    ),
)
def _calc_mag_moment_basic_with_jacobian_radial_coeffs(
    itype,
    jtypes,
    r_abs: np.ndarray,
    r_ijs_unit: np.ndarray,
    alpha_index_basic: np.ndarray,
    rb_values: np.ndarray,
    rb_derivs: np.ndarray,
    drb_dmis: np.ndarray,
    drb_dmjs: np.ndarray,
    rb_coeffs: np.ndarray,
    moment_values: np.ndarray,
    moment_jac_rs: np.ndarray,
    moment_jac_mis: np.ndarray,
    moment_jac_mjs: np.ndarray,
    moment_jac_cs: np.ndarray,
    moment_jac_rc: np.ndarray,
    moment_jac_mic: np.ndarray,
    moment_jac_mjc: np.ndarray,
) -> None:
    """Compute basic moment components and its jacobian wrt `r_ijs`."""
    # Precompute powers
    max_pow = int(np.max(alpha_index_basic))
    r_unit_pows = _calc_r_unit_pows(r_ijs_unit, max_pow)

    rbs = rb_coeffs.shape[3]
    der = np.zeros(3)
    for i_aib, aib in enumerate(alpha_index_basic):
        mu, xpow, ypow, zpow = aib
        xyzpow = xpow + ypow + zpow
        for j, jtype in enumerate(jtypes):
            mult0 = 1.0
            mult0 *= r_unit_pows[xpow, j, 0]
            mult0 *= r_unit_pows[ypow, j, 1]
            mult0 *= r_unit_pows[zpow, j, 2]
            for i_rb in range(rbs):
                val = rb_values[i_rb, j] * mult0
                for k in range(3):
                    der[k] = (
                        r_ijs_unit[j, k]
                        * mult0
                        * (rb_derivs[i_rb, j] - xyzpow * rb_values[i_rb, j] / r_abs[j])
                    )
                if xpow != 0:
                    der[0] += (
                        rb_values[i_rb, j]
                        * (xpow * r_unit_pows[xpow - 1, j, 0])
                        * r_unit_pows[ypow, j, 1]
                        * r_unit_pows[zpow, j, 2]
                        / r_abs[j]
                    )
                if ypow != 0:
                    der[1] += (
                        rb_values[i_rb, j]
                        * r_unit_pows[xpow, j, 0]
                        * (ypow * r_unit_pows[ypow - 1, j, 1])
                        * r_unit_pows[zpow, j, 2]
                        / r_abs[j]
                    )
                if zpow != 0:
                    der[2] += (
                        rb_values[i_rb, j]
                        * r_unit_pows[xpow, j, 0]
                        * r_unit_pows[ypow, j, 1]
                        * (zpow * r_unit_pows[zpow - 1, j, 2])
                        / r_abs[j]
                    )
                c = rb_coeffs[itype, jtype, mu, i_rb]

                moment_values[i_aib] += c * val
                moment_jac_mis[i_aib, j] += c * mult0 * drb_dmis[i_rb, j]
                moment_jac_mjs[i_aib, j] += c * mult0 * drb_dmjs[i_rb, j]
                for k in range(3):
                    moment_jac_rs[i_aib, j, k] += c * der[k]
                moment_jac_cs[i_aib, jtype, mu, i_rb] += val
                moment_jac_mic[i_aib, jtype, mu, i_rb, j] += mult0 * drb_dmis[i_rb, j]
                moment_jac_mjc[i_aib, jtype, mu, i_rb, j] += mult0 * drb_dmjs[i_rb, j]
                for k in range(3):
                    moment_jac_rc[i_aib, jtype, mu, i_rb, j, k] += der[k]


@nb.njit(
    nb.types.Tuple(
        (
            nb.float64[:, :, :],
            nb.float64[:, :, :, :, :],
            nb.float64[:, :, :, :],
            nb.float64[:, :, :, :],
        ),
    )(
        nb.int32,
        nb.int32[:, :],
        nb.int32[:],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:, :, :],
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64[:, :, :, :],
        nb.float64[:, :, :, :, :, :],
        nb.float64[:, :, :, :, :],
        nb.float64[:, :, :, :, :],
    ),
)
def _calc_mag_dedcs_and_dgdcs(
    alpha_index_basic_count: np.int32,
    alpha_index_times: np.ndarray,
    alpha_moment_mapping: np.ndarray,
    moment_coeffs: np.ndarray,
    moment_values: np.ndarray,
    moment_jac_rs: np.ndarray,
    moment_jac_mis: np.ndarray,
    moment_jac_mjs: np.ndarray,
    moment_jac_cs: np.ndarray,
    moment_jac_rc: np.ndarray,
    moment_jac_mic: np.ndarray,
    moment_jac_mjc: np.ndarray,
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
    dgmidcs : np.ndarray
        d(dV/dm_i)/dc.
    dgmjdcs : np.ndarray
        d(dV/dm_j)/dc.

    """
    _, spc, rfc, rbs, nns, _ = moment_jac_rc.shape

    dedmb = np.zeros_like(moment_values)
    dgdmb = np.zeros_like(moment_jac_rs)
    dgmidmb = np.zeros_like(moment_jac_mis)
    dgmjdmb = np.zeros_like(moment_jac_mis)
    dedmb[alpha_moment_mapping] = moment_coeffs  # dV/dB
    for ait in alpha_index_times[::-1]:
        i1, i2, mult, i3 = ait
        dedmb[i1] += mult * dedmb[i3] * moment_values[i2]
        dedmb[i2] += mult * dedmb[i3] * moment_values[i1]
    for ait in alpha_index_times:
        i1, i2, mult, i3 = ait
        for j in range(nns):
            dgmidmb[i1, j] += mult * dedmb[i3] * moment_jac_mis[i2, j]
            dgmidmb[i2, j] += mult * dedmb[i3] * moment_jac_mis[i1, j]
            dgmjdmb[i1, j] += mult * dedmb[i3] * moment_jac_mjs[i2, j]
            dgmjdmb[i2, j] += mult * dedmb[i3] * moment_jac_mjs[i1, j]
            for ixyz0 in range(3):
                dgdmb[i1, j, ixyz0] += mult * dedmb[i3] * moment_jac_rs[i2, j, ixyz0]
                dgdmb[i2, j, ixyz0] += mult * dedmb[i3] * moment_jac_rs[i1, j, ixyz0]
    for ait in alpha_index_times[::-1]:
        i1, i2, mult, i3 = ait
        for j in range(nns):
            dgmidmb[i1, j] += mult * dgmidmb[i3, j] * moment_values[i2]
            dgmidmb[i2, j] += mult * dgmidmb[i3, j] * moment_values[i1]
            dgmjdmb[i1, j] += mult * dgmjdmb[i3, j] * moment_values[i2]
            dgmjdmb[i2, j] += mult * dgmjdmb[i3, j] * moment_values[i1]
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
    dgmidcs = np.zeros(moment_jac_mic.shape[1:])
    dgmjdcs = np.zeros(moment_jac_mic.shape[1:])
    for iamc in range(alpha_index_basic_count):
        v1 = dedmb[iamc]
        for ispc in range(spc):
            for irfc in range(rfc):
                for irbs in range(rbs):
                    v3 = moment_jac_cs[iamc, ispc, irfc, irbs]
                    for j in range(nns):
                        vm2 = moment_jac_mic[iamc, ispc, irfc, irbs, j]
                        dgmidcs[ispc, irfc, irbs, j] += vm2 * v1
                        dgmidcs[ispc, irfc, irbs, j] += v3 * dgmidmb[iamc, j]
                        vm2 = moment_jac_mjc[iamc, ispc, irfc, irbs, j]
                        dgmjdcs[ispc, irfc, irbs, j] += vm2 * v1
                        dgmjdcs[ispc, irfc, irbs, j] += v3 * dgmjdmb[iamc, j]
                        for ixyz in range(3):
                            v2 = moment_jac_rc[iamc, ispc, irfc, irbs, j, ixyz]
                            v4 = dgdmb[iamc, j, ixyz]
                            dgdcs[ispc, irfc, irbs, j, ixyz] += v2 * v1
                            dgdcs[ispc, irfc, irbs, j, ixyz] += v3 * v4

    return dedcs, dgdcs, dgmidcs, dgmjdcs


@nb.njit
def store_mag_radial_basis(
    i: np.int32,
    itype: np.int32,
    js: np.ndarray,
    jtypes: np.ndarray,
    r_ijs: np.ndarray,
    r_ijs_unit: np.ndarray,
    basis_vs: np.ndarray,
    basis_ds: np.ndarray,
    basis_dmis: np.ndarray,
    basis_dmjs: np.ndarray,
    values: np.ndarray,
    dqdris: np.ndarray,
    dqdmis: np.ndarray,
    dqdeps: np.ndarray,
) -> None:
    radial_basis_size = basis_vs.shape[0]
    for ib in range(radial_basis_size):
        for k, j in enumerate(js):
            jtype = jtypes[k]
            values[itype, jtype, ib] += basis_vs[ib, k]
            dqdmis[itype, jtype, ib, i] += basis_dmis[ib, k]
            dqdmis[itype, jtype, ib, j] += basis_dmjs[ib, k]
            for ixyz0 in range(3):
                tmp = basis_ds[ib, k] * r_ijs_unit[k, ixyz0]
                dqdris[itype, jtype, ib, i, ixyz0] -= tmp
                dqdris[itype, jtype, ib, j, ixyz0] += tmp
                for ixyz1 in range(3):
                    dqdeps[itype, jtype, ib, ixyz0, ixyz1] += tmp * r_ijs[k, ixyz1]


@nb.njit(
    (
        nb.float64[:],
        nb.float64[:, :],
        nb.int32[:, :],
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64[:],
        nb.float64[:, :, :],
        nb.float64[:, :],
        nb.float64[:, :],
    ),
)
def _calc_mag_moment_basic(
    r_abs,
    r_ijs_unit,
    alpha_index_basic,
    rb_values,
    drb_drs,
    drb_dmis,
    drb_dmjs,
    moment_values,
    moment_jac_rs,
    moment_jac_mis,
    moment_jac_mjs,
) -> None:
    """Compute basic moment components and its jacobian wrt `r_ijs`."""
    # Precompute powers
    max_pow = int(np.max(alpha_index_basic))
    r_unit_pows = _calc_r_unit_pows(r_ijs_unit, max_pow)

    for i_aib, aib in enumerate(alpha_index_basic):
        mu, xpow, ypow, zpow = aib
        xyzpow = xpow + ypow + zpow
        for j in range(r_abs.size):
            mult0 = 1.0
            mult0 *= r_unit_pows[xpow, j, 0]
            mult0 *= r_unit_pows[ypow, j, 1]
            mult0 *= r_unit_pows[zpow, j, 2]
            moment_values[i_aib] += rb_values[mu, j] * mult0
            moment_jac_mis[i_aib, j] = drb_dmis[mu, j] * mult0
            moment_jac_mjs[i_aib, j] = drb_dmjs[mu, j] * mult0
            for k in range(3):
                moment_jac_rs[i_aib, j, k] = (
                    r_ijs_unit[j, k]
                    * mult0
                    * (drb_drs[mu, j] - xyzpow * rb_values[mu, j] / r_abs[j])
                )
            if xpow != 0:
                moment_jac_rs[i_aib, j, 0] += (
                    rb_values[mu, j]
                    * (xpow * r_unit_pows[xpow - 1, j, 0])
                    * r_unit_pows[ypow, j, 1]
                    * r_unit_pows[zpow, j, 2]
                    / r_abs[j]
                )
            if ypow != 0:
                moment_jac_rs[i_aib, j, 1] += (
                    rb_values[mu, j]
                    * r_unit_pows[xpow, j, 0]
                    * (ypow * r_unit_pows[ypow - 1, j, 1])
                    * r_unit_pows[zpow, j, 2]
                    / r_abs[j]
                )
            if zpow != 0:
                moment_jac_rs[i_aib, j, 2] += (
                    rb_values[mu, j]
                    * r_unit_pows[xpow, j, 0]
                    * r_unit_pows[ypow, j, 1]
                    * (zpow * r_unit_pows[zpow - 1, j, 2])
                    / r_abs[j]
                )


@nb.njit(
    nb.float64[:](
        nb.float64[:, :],
        nb.float64[:],
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64[:, :],
        nb.int32,
        nb.int32[:],
        nb.int32[:, :],
        nb.int32[:, :],
        nb.float64[:],
        nb.float64[:, :],
        nb.float64[:],
        nb.float64[:],
    ),
)
def calc_mag_moments_run(
    r_ijs_unit: npt.NDArray[np.float64],
    r_abs: npt.NDArray[np.float64],
    rb_vals: npt.NDArray[np.float64],
    drb_drs: npt.NDArray[np.float64],
    drb_dmis: npt.NDArray[np.float64],
    drb_dmjs: npt.NDArray[np.float64],
    alpha_moments_count: np.int32,
    alpha_moment_mapping: npt.NDArray[np.int32],
    alpha_index_basic: npt.NDArray[np.int32],
    alpha_index_times: npt.NDArray[np.int32],
    moment_coeffs: npt.NDArray[np.float64],
    grad: npt.NDArray[np.float64],
    grad_mag_i: npt.NDArray[np.float64],
    grad_mag_j: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    amc = alpha_moments_count
    njs = r_abs.size
    moment_values = np.zeros(amc)
    moment_jac_r = np.zeros((amc, njs, 3))
    moment_jac_mis = np.zeros((amc, njs))
    moment_jac_mjs = np.zeros((amc, njs))

    _calc_mag_moment_basic(
        r_abs,
        r_ijs_unit,
        alpha_index_basic,
        rb_vals,
        drb_drs,
        drb_dmis,
        drb_dmjs,
        moment_values,
        moment_jac_r,
        moment_jac_mis,
        moment_jac_mjs,
    )

    _mag_contract_moments_backwards(
        alpha_index_basic,
        alpha_index_times,
        alpha_moment_mapping,
        moment_coeffs,
        moment_values,
        moment_jac_r,
        moment_jac_mis,
        moment_jac_mjs,
        grad,
        grad_mag_i,
        grad_mag_j,
    )

    return moment_values[alpha_moment_mapping]


@nb.njit(
    nb.types.Tuple(
        (
            nb.float64[:],
            nb.float64[:, :, :],
            nb.float64[:, :],
            nb.float64[:, :],
            nb.float64[:, :, :],
            nb.float64[:, :, :, :, :],
            nb.float64[:, :, :, :],
            nb.float64[:, :, :, :],
        ),
    )(
        nb.int32,
        nb.int32[:],
        nb.float64[:],
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64[:, :, :, :],
        nb.int32,
        nb.int32[:],
        nb.int32[:, :],
        nb.int32[:, :],
        nb.float64[:],
    ),
)
def calc_mag_moments_train(
    itype,
    jtypes,
    r_abs: np.ndarray,
    r_ijs_unit: np.ndarray,
    rb_values: np.ndarray,
    drb_drijs: np.ndarray,
    drb_dmis: np.ndarray,
    drb_dmjs: np.ndarray,
    rb_coeffs: np.ndarray,
    alpha_moments_count: np.int32,
    alpha_moment_mapping: np.ndarray,
    alpha_index_basic: np.ndarray,
    alpha_index_times: np.ndarray,
    moment_coeffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, species_count, rfs, rbs = rb_coeffs.shape
    njs, _ = r_ijs_unit.shape
    amc = alpha_moments_count
    moment_values = np.zeros(amc)
    moment_jac_rs = np.zeros((amc, njs, 3))
    moment_jac_mis = np.zeros((amc, njs))
    moment_jac_mjs = np.zeros((amc, njs))
    moment_jac_cs = np.zeros((amc, species_count, rfs, rbs))
    moment_jac_rc = np.zeros((amc, species_count, rfs, rbs, njs, 3))
    moment_jac_mic = np.zeros((amc, species_count, rfs, rbs, njs))
    moment_jac_mjc = np.zeros((amc, species_count, rfs, rbs, njs))

    _calc_mag_moment_basic_with_jacobian_radial_coeffs(
        itype,
        jtypes,
        r_abs,
        r_ijs_unit,
        alpha_index_basic,
        rb_values,
        drb_drijs,
        drb_dmis,
        drb_dmjs,
        rb_coeffs,
        moment_values,
        moment_jac_rs,
        moment_jac_mis,
        moment_jac_mjs,
        moment_jac_cs,
        moment_jac_rc,
        moment_jac_mic,
        moment_jac_mjc,
    )

    _mag_contract_moments(
        alpha_index_times,
        moment_values,
        moment_jac_rs,
        moment_jac_mis,
        moment_jac_mjs,
    )

    dedcs, dgdcs, dgmidcs, dgmjdcs = _calc_mag_dedcs_and_dgdcs(
        alpha_index_basic.shape[0],
        alpha_index_times,
        alpha_moment_mapping,
        moment_coeffs,
        moment_values,
        moment_jac_rs,
        moment_jac_mis,
        moment_jac_mjs,
        moment_jac_cs,
        moment_jac_rc,
        moment_jac_mic,
        moment_jac_mjc,
    )

    return (
        moment_values[alpha_moment_mapping],
        moment_jac_rs[alpha_moment_mapping],
        moment_jac_mis[alpha_moment_mapping],
        moment_jac_mjs[alpha_moment_mapping],
        dedcs,
        dgdcs,
        dgmidcs,
        dgmjdcs,
    )
