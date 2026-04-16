"""MTP written using numba."""

import numba as nb
import numpy as np
import numpy.typing as npt
from ase import Atoms

from motep.potentials.mmtp.base import MagEngineBase
from motep.potentials.mtp.data import get_types
from motep.potentials.mtp.numba.chebyshev import sum_radial_terms
from motep.potentials.mtp.numba.engine import (
    _calc_forces_from_gradient,
    _calc_r_unit,
    _nb_linalg_norm,
)
from motep.potentials.mtp.numba.moment import (
    calc_moments_train,
    store_radial_basis,
    update_mbd_dbdeps,
    update_mbd_dbdris,
    update_mbd_dgdcs,
    update_mbd_dsdcs,
    update_mbd_dvdcs,
    update_mbd_vatoms,
)

from .chebyshev import calc_radial_and_mag_basis
from .moment import (
    calc_mag_moments_run,
    calc_mag_moments_train,
    store_mag_radial_basis,
    update_mbd_dbdmis,
    update_mbd_dgmdcs,
)


class NumbaMagMTPEngine(MagEngineBase):
    """MTP Engine based on Numba."""

    def _calculate(self, atoms: Atoms, magmoms: np.ndarray | None = None) -> tuple:
        if magmoms is None:
            magmoms = atoms.get_initial_magnetic_moments()
        if self.mode == "run":
            return self._calc_mag_run(atoms, magmoms)
        if self.mode == "train":
            return self._calc_mag_train(atoms, magmoms)
        if self.mode == "train_mgrad":
            return self._calc_mag_train_mgrad(atoms, magmoms)
        raise NotImplementedError(self.mode)

    def _calc_mag_run(self, atoms: Atoms, magmoms: np.ndarray) -> tuple:
        mtp_data = self.mtp_data

        js = self._neighbors
        rs = self._get_interatomic_vectors(atoms)

        magnetic_moments = magmoms

        itypes = get_types(atoms, mtp_data.species)
        jtypes = itypes[js]

        self.mbd.clean()
        self.rbd.clean()

        energies, lgrads, mgrad_i, mgrad_j = _calc_mag_run(
            js,
            rs,
            magnetic_moments,
            itypes,
            jtypes,
            mtp_data.scaling,
            mtp_data.magnetic_basis.min,
            mtp_data.magnetic_basis.max,
            mtp_data.magnetic_basis.size,
            mtp_data.radial_basis.min,
            mtp_data.radial_basis.max,
            mtp_data.radial_basis.size,
            mtp_data.alpha_moments_count,
            mtp_data.alpha_index_basic,
            mtp_data.alpha_index_times,
            mtp_data.alpha_moment_mapping,
            mtp_data.radial_coeffs,
            mtp_data.species_coeffs,
            mtp_data.moment_coeffs,
            self.mbd.vatoms,
        )

        forces = _calc_forces_from_gradient(lgrads, js)
        stress = np.einsum("ijk, ijl -> lk", rs, lgrads)
        mgrad = _calc_mgrad_from_gradient(mgrad_i, mgrad_j, js)

        return energies, forces, stress, mgrad

    def _calc_mag_train(self, atoms: Atoms, magmoms: np.ndarray) -> tuple:
        mtp_data = self.mtp_data

        js = self._neighbors
        rs = self._get_interatomic_vectors(atoms)

        magnetic_moments = magmoms

        itypes = get_types(atoms, mtp_data.species)
        jtypes = itypes[js]

        self.mbd.clean()
        self.rbd.clean()

        energies = _calc_mag_train(
            js,
            rs,
            magnetic_moments,
            itypes,
            jtypes,
            mtp_data.scaling,
            mtp_data.magnetic_basis.min,
            mtp_data.magnetic_basis.max,
            mtp_data.magnetic_basis.size,
            mtp_data.radial_basis.min,
            mtp_data.radial_basis.max,
            mtp_data.radial_basis.size,
            mtp_data.alpha_moments_count,
            mtp_data.alpha_index_basic,
            mtp_data.alpha_index_times,
            mtp_data.alpha_moment_mapping,
            mtp_data.radial_coeffs,
            mtp_data.species_coeffs,
            mtp_data.moment_coeffs,
            self.rbd.values,
            self.rbd.dqdris,
            self.rbd.dqdeps,
            self.mbd.vatoms,
            self.mbd.dbdris,
            self.mbd.dbdeps,
            self.mbd.dedcs,
            self.mbd.dgdcs,
            self.mbd.dsdcs,
        )

        # Use the run implementation to get the magnetic gradients.
        # This is cheap compared to the train call.
        _, _, mgrad_i, mgrad_j = _calc_mag_run(
            js,
            rs,
            magnetic_moments,
            itypes,
            jtypes,
            mtp_data.scaling,
            mtp_data.magnetic_basis.min,
            mtp_data.magnetic_basis.max,
            mtp_data.magnetic_basis.size,
            mtp_data.radial_basis.min,
            mtp_data.radial_basis.max,
            mtp_data.radial_basis.size,
            mtp_data.alpha_moments_count,
            mtp_data.alpha_index_basic,
            mtp_data.alpha_index_times,
            mtp_data.alpha_moment_mapping,
            mtp_data.radial_coeffs,
            mtp_data.species_coeffs,
            mtp_data.moment_coeffs,
            self.mbd.vatoms.copy(),  # Don't accumulate extra mbd values
        )

        moment_coeffs = mtp_data.moment_coeffs

        forces = np.sum(moment_coeffs * self.mbd.dbdris.T, axis=-1).T * -1.0
        stress = np.sum(moment_coeffs * self.mbd.dbdeps.T, axis=-1).T
        mgrad = _calc_mgrad_from_gradient(mgrad_i, mgrad_j, js)

        return energies, forces, stress, mgrad

    def _calc_mag_train_mgrad(self, atoms: Atoms, magmoms: np.ndarray) -> tuple:
        mtp_data = self.mtp_data

        js = self._neighbors
        rs = self._get_interatomic_vectors(atoms)

        magnetic_moments = magmoms

        itypes = get_types(atoms, mtp_data.species)
        jtypes = itypes[js]

        self.mbd.clean()
        self.rbd.clean()

        energies = _calc_mag_train_mgrad(
            js,
            rs,
            magnetic_moments,
            itypes,
            jtypes,
            mtp_data.scaling,
            mtp_data.magnetic_basis.min,
            mtp_data.magnetic_basis.max,
            mtp_data.magnetic_basis.size,
            mtp_data.radial_basis.min,
            mtp_data.radial_basis.max,
            mtp_data.radial_basis.size,
            mtp_data.alpha_moments_count,
            mtp_data.alpha_index_basic,
            mtp_data.alpha_index_times,
            mtp_data.alpha_moment_mapping,
            mtp_data.radial_coeffs,
            mtp_data.species_coeffs,
            mtp_data.moment_coeffs,
            self.rbd.values,
            self.rbd.dqdris,
            self.rbd.dqdmis,
            self.rbd.dqdeps,
            self.mbd.vatoms,
            self.mbd.dbdris,
            self.mbd.dbdmis,
            self.mbd.dbdeps,
            self.mbd.dedcs,
            self.mbd.dgdcs,
            self.mbd.dgmdcs,
            self.mbd.dsdcs,
        )

        moment_coeffs = mtp_data.moment_coeffs

        forces = np.sum(moment_coeffs * self.mbd.dbdris.T, axis=-1).T * -1.0
        mgrad = np.sum(moment_coeffs * self.mbd.dbdmis.T, axis=-1).T
        stress = np.sum(moment_coeffs * self.mbd.dbdeps.T, axis=-1).T

        return energies, forces, stress, mgrad


@nb.njit(
    nb.float64[:](
        nb.float64[:, :],
        nb.float64[:, :],
        nb.int32[:, :],
    ),
)
def _calc_mgrad_from_gradient(
    grad_i: np.ndarray,
    grad_j: np.ndarray,
    js: np.ndarray,
) -> np.ndarray:
    nis, njs = js.shape
    mag_grad = np.zeros(nis)
    for i in range(nis):
        for i_j in range(njs):
            j = js[i, i_j]
            mag_grad[i] += grad_i[i, i_j]
            mag_grad[j] += grad_j[i, i_j]
    return mag_grad


@nb.njit(
    nb.types.Tuple(
        (
            nb.float64[:],
            nb.float64[:, :, :],
            nb.float64[:, :],
            nb.float64[:, :],
        ),
    )(
        nb.int32[:, :],
        nb.float64[:, :, :],
        nb.float64[:],
        nb.int32[:],
        nb.int32[:, :],
        nb.float64,
        nb.float64,
        nb.float64,
        nb.int32,
        nb.float64,
        nb.float64,
        nb.int32,
        nb.int32,
        nb.int32[:, :],
        nb.int32[:, :],
        nb.int32[:],
        nb.float64[:, :, :, :],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:, :],
    ),
    # parallel=True,
    cache=True,
)
def _calc_mag_run(
    js: npt.NDArray[np.int32],
    rs: npt.NDArray[np.float64],
    magnetic_moments: npt.NDArray[np.float64],
    itypes: npt.NDArray[np.int32],
    all_jtypes: npt.NDArray[np.int32],
    scaling: np.float64,
    min_mag: np.float64,
    max_mag: np.float64,
    mag_basis_size: np.int32,
    min_dist: np.float64,
    max_dist: np.float64,
    rad_basis_size: np.int32,
    alpha_moments_count: np.int32,
    alpha_index_basic: npt.NDArray[np.int32],
    alpha_index_times: npt.NDArray[np.int32],
    alpha_moment_mapping: npt.NDArray[np.int32],
    radial_coeffs: npt.NDArray[np.float64],
    species_coeffs: npt.NDArray[np.float64],
    moment_coeffs: npt.NDArray[np.float64],
    mbd_vatoms: npt.NDArray[np.float64],
):
    energies = species_coeffs[itypes]
    gradient = np.zeros((itypes.size, all_jtypes.shape[1], 3))
    grad_mag_i = np.zeros((itypes.size, all_jtypes.shape[1]))
    grad_mag_j = np.zeros((itypes.size, all_jtypes.shape[1]))
    mb_vals = np.empty((itypes.size, alpha_moment_mapping.size))
    for i in nb.prange(itypes.size):
        js_i = js[i, :]
        rs_i = rs[i, :, :]
        ms = magnetic_moments[js_i]
        jtypes_i = all_jtypes[i, :]
        r_abs = _nb_linalg_norm(rs_i)
        r_unit = _calc_r_unit(rs_i, r_abs)
        rb_vals, drb_drs, drb_dmis, drb_dmjs = calc_radial_and_mag_basis(
            r_abs,
            ms,
            rad_basis_size,
            mag_basis_size,
            scaling,
            min_dist,
            max_dist,
            magnetic_moments[i],
            min_mag,
            max_mag,
        )
        rfuncs = sum_radial_terms(itypes[i], jtypes_i, rb_vals, radial_coeffs)
        drf_drs = sum_radial_terms(itypes[i], jtypes_i, drb_drs, radial_coeffs)
        drf_dmis = sum_radial_terms(itypes[i], jtypes_i, drb_dmis, radial_coeffs)
        drf_dmjs = sum_radial_terms(itypes[i], jtypes_i, drb_dmjs, radial_coeffs)
        mb_values = calc_mag_moments_run(
            r_unit,
            r_abs,
            rfuncs,
            drf_drs,
            drf_dmis,
            drf_dmjs,
            alpha_moments_count,
            alpha_moment_mapping,
            alpha_index_basic,
            alpha_index_times,
            moment_coeffs,
            gradient[i, :, :],
            grad_mag_i[i, :],
            grad_mag_j[i, :],
        )

        mb_vals[i] = mb_values

        for basis_i, coeff in enumerate(moment_coeffs):
            energies[i] += coeff * mb_values[basis_i]

    for i in range(itypes.size):
        update_mbd_vatoms(i, mbd_vatoms, mb_vals[i])

    return energies, gradient, grad_mag_i, grad_mag_j


@nb.njit(
    nb.float64[:](
        nb.int32[:, :],
        nb.float64[:, :, :],
        nb.float64[:],
        nb.int32[:],
        nb.int32[:, :],
        nb.float64,
        nb.float64,
        nb.float64,
        nb.int32,
        nb.float64,
        nb.float64,
        nb.int32,
        nb.int32,
        nb.int32[:, :],
        nb.int32[:, :],
        nb.int32[:],
        nb.float64[:, :, :, :],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:, :, :],
        nb.float64[:, :, :, :, :],
        nb.float64[:, :, :, :, :],
        nb.float64[:, :],
        nb.float64[:, :, :],
        nb.float64[:, :, :],
        nb.float64[:, :, :, :],
        nb.float64[:, :, :, :, :, :],
        nb.float64[:, :, :, :, :, :],
    ),
    # parallel=True,
    cache=True,
)
def _calc_mag_train(
    js: npt.NDArray[np.int32],
    rs: npt.NDArray[np.float64],
    magnetic_moments: npt.NDArray[np.float64],
    itypes: npt.NDArray[np.int32],
    jtypes: npt.NDArray[np.int32],
    scaling: np.float64,
    min_mag: np.float64,
    max_mag: np.float64,
    mag_basis_size: np.int32,
    min_dist: np.float64,
    max_dist: np.float64,
    rad_basis_size: np.int32,
    alpha_moments_count: np.int32,
    alpha_index_basic: npt.NDArray[np.int32],
    alpha_index_times: npt.NDArray[np.int32],
    alpha_moment_mapping: npt.NDArray[np.int32],
    radial_coeffs: npt.NDArray[np.float64],
    species_coeffs: npt.NDArray[np.float64],
    moment_coeffs: npt.NDArray[np.float64],
    rbd_values: npt.NDArray[np.float64],
    rbd_dqdris: npt.NDArray[np.float64],
    rbd_dqdeps: npt.NDArray[np.float64],
    mbd_vatoms: npt.NDArray[np.float64],
    mbd_dbdris: npt.NDArray[np.float64],
    mbd_dbdeps: npt.NDArray[np.float64],
    mbd_dedcs: npt.NDArray[np.float64],
    mbd_dgdcs: npt.NDArray[np.float64],
    mbd_dsdcs: npt.NDArray[np.float64],
):
    _, species_count, rfs, rbs = radial_coeffs.shape
    nis, njs, _ = rs.shape
    rb_vals = np.empty((nis, radial_coeffs.shape[3], njs))
    drb_drs = np.empty((nis, radial_coeffs.shape[3], njs))
    mb_vals = np.empty((nis, alpha_moment_mapping.size))
    mb_jac_rs = np.empty((nis, alpha_moment_mapping.size, njs, 3))
    dedcs = np.empty((nis, species_count, rfs, rbs))
    dgdcs = np.empty((nis, species_count, rfs, rbs, njs, 3))

    energies = species_coeffs[itypes]
    for i in nb.prange(itypes.size):
        js_i = js[i, :]
        rs_i = rs[i, :, :]
        ms = magnetic_moments[js_i]
        jtypes_i = jtypes[i, :]
        r_abs = _nb_linalg_norm(rs_i)
        r_unit = _calc_r_unit(rs_i, r_abs)
        rb_vals_i, drb_drs_i, drb_dmis_i, drb_dmjs_i = calc_radial_and_mag_basis(
            r_abs,
            ms,
            rad_basis_size,
            mag_basis_size,
            scaling,
            min_dist,
            max_dist,
            magnetic_moments[i],
            min_mag,
            max_mag,
        )
        (mb_vals_i, mb_jac_rs_i, dedcs_i, dgdcs_i) = calc_moments_train(
            itypes[i],
            jtypes_i,
            r_abs,
            r_unit,
            rb_vals_i,
            drb_drs_i,
            radial_coeffs,
            alpha_moments_count,
            alpha_moment_mapping,
            alpha_index_basic,
            alpha_index_times,
            moment_coeffs,
        )

        for basis_i, coeff in enumerate(moment_coeffs):
            energies[i] += coeff * mb_vals_i[basis_i]

        rb_vals[i] = rb_vals_i
        drb_drs[i] = drb_drs_i
        mb_vals[i] = mb_vals_i
        mb_jac_rs[i] = mb_jac_rs_i
        dedcs[i] = dedcs_i
        dgdcs[i] = dgdcs_i

    for i in range(itypes.size):
        js_i = js[i, :]
        rs_i = rs[i, :, :]
        jtypes_i = jtypes[i, :]
        r_abs = _nb_linalg_norm(rs_i)
        r_unit = _calc_r_unit(rs_i, r_abs)
        store_radial_basis(
            i,
            itypes[i],
            js_i,
            jtypes_i,
            rs_i,
            r_unit,
            rb_vals[i],
            drb_drs[i],
            rbd_values,
            rbd_dqdris,
            rbd_dqdeps,
        )
        update_mbd_vatoms(i, mbd_vatoms, mb_vals[i])
        update_mbd_dbdris(i, js_i, mbd_dbdris, mb_jac_rs[i])
        update_mbd_dbdeps(js_i, rs_i, mbd_dbdeps, mb_jac_rs[i])
        update_mbd_dvdcs(i, itypes[i], mbd_dedcs, dedcs[i])
        update_mbd_dgdcs(i, itypes[i], js_i, mbd_dgdcs, dgdcs[i])
        update_mbd_dsdcs(itypes[i], js_i, rs_i, mbd_dsdcs, dgdcs[i])

    return energies


@nb.njit(
    nb.float64[:](
        nb.int32[:, :],
        nb.float64[:, :, :],
        nb.float64[:],
        nb.int32[:],
        nb.int32[:, :],
        nb.float64,
        nb.float64,
        nb.float64,
        nb.int32,
        nb.float64,
        nb.float64,
        nb.int32,
        nb.int32,
        nb.int32[:, :],
        nb.int32[:, :],
        nb.int32[:],
        nb.float64[:, :, :, :],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:, :, :],
        nb.float64[:, :, :, :, :],
        nb.float64[:, :, :, :],
        nb.float64[:, :, :, :, :],
        nb.float64[:, :],
        nb.float64[:, :, :],
        nb.float64[:, :],
        nb.float64[:, :, :],
        nb.float64[:, :, :, :],
        nb.float64[:, :, :, :, :, :],
        nb.float64[:, :, :, :, :],
        nb.float64[:, :, :, :, :, :],
    ),
    # parallel=True,
    cache=True,
)
def _calc_mag_train_mgrad(
    js: npt.NDArray[np.int32],
    rs: npt.NDArray[np.float64],
    magnetic_moments: npt.NDArray[np.float64],
    itypes: npt.NDArray[np.int32],
    jtypes: npt.NDArray[np.int32],
    scaling: np.float64,
    min_mag: np.float64,
    max_mag: np.float64,
    mag_basis_size: np.int32,
    min_dist: np.float64,
    max_dist: np.float64,
    rad_basis_size: np.int32,
    alpha_moments_count: np.int32,
    alpha_index_basic: npt.NDArray[np.int32],
    alpha_index_times: npt.NDArray[np.int32],
    alpha_moment_mapping: npt.NDArray[np.int32],
    radial_coeffs: npt.NDArray[np.float64],
    species_coeffs: npt.NDArray[np.float64],
    moment_coeffs: npt.NDArray[np.float64],
    rbd_values: npt.NDArray[np.float64],
    rbd_dqdris: npt.NDArray[np.float64],
    rbd_dqdmis: npt.NDArray[np.float64],
    rbd_dqdeps: npt.NDArray[np.float64],
    mbd_vatoms: npt.NDArray[np.float64],
    mbd_dbdris: npt.NDArray[np.float64],
    mbd_dbdmis: npt.NDArray[np.float64],
    mbd_dbdeps: npt.NDArray[np.float64],
    mbd_dedcs: npt.NDArray[np.float64],
    mbd_dgdcs: npt.NDArray[np.float64],
    mbd_dgmdcs: npt.NDArray[np.float64],
    mbd_dsdcs: npt.NDArray[np.float64],
):
    _, species_count, rfs, rbs = radial_coeffs.shape
    nis, njs, _ = rs.shape
    rb_vals = np.empty((nis, radial_coeffs.shape[3], njs))
    drb_drs = np.empty((nis, radial_coeffs.shape[3], njs))
    drb_dmis = np.empty((nis, radial_coeffs.shape[3], njs))
    drb_dmjs = np.empty((nis, radial_coeffs.shape[3], njs))
    mb_vals = np.empty((nis, alpha_moment_mapping.size))
    mb_jac_rs = np.empty((nis, alpha_moment_mapping.size, njs, 3))
    mb_jac_mis = np.empty((nis, alpha_moment_mapping.size, njs))
    mb_jac_mjs = np.empty((nis, alpha_moment_mapping.size, njs))
    dedcs = np.empty((nis, species_count, rfs, rbs))
    dgdcs = np.empty((nis, species_count, rfs, rbs, njs, 3))
    dgmidcs = np.empty((nis, species_count, rfs, rbs, njs))
    dgmjdcs = np.empty((nis, species_count, rfs, rbs, njs))

    energies = species_coeffs[itypes]
    for i in nb.prange(itypes.size):
        js_i = js[i, :]
        rs_i = rs[i, :, :]
        ms = magnetic_moments[js_i]
        jtypes_i = jtypes[i, :]
        r_abs = _nb_linalg_norm(rs_i)
        r_unit = _calc_r_unit(rs_i, r_abs)
        rb_vals_i, drb_drs_i, drb_dmis_i, drb_dmjs_i = calc_radial_and_mag_basis(
            r_abs,
            ms,
            rad_basis_size,
            mag_basis_size,
            scaling,
            min_dist,
            max_dist,
            magnetic_moments[i],
            min_mag,
            max_mag,
        )
        (
            mb_vals_i,
            mb_jac_rs_i,
            mb_jac_mis_i,
            mb_jac_mjs_i,
            dedcs_i,
            dgdcs_i,
            dgmidcs_i,
            dgmjdcs_i,
        ) = calc_mag_moments_train(
            itypes[i],
            jtypes_i,
            r_abs,
            r_unit,
            rb_vals_i,
            drb_drs_i,
            drb_dmis_i,
            drb_dmjs_i,
            radial_coeffs,
            alpha_moments_count,
            alpha_moment_mapping,
            alpha_index_basic,
            alpha_index_times,
            moment_coeffs,
        )

        for basis_i, coeff in enumerate(moment_coeffs):
            energies[i] += coeff * mb_vals_i[basis_i]

        rb_vals[i] = rb_vals_i
        drb_drs[i] = drb_drs_i
        drb_dmis[i] = drb_dmis_i
        drb_dmjs[i] = drb_dmjs_i
        mb_vals[i] = mb_vals_i
        mb_jac_rs[i] = mb_jac_rs_i
        mb_jac_mis[i] = mb_jac_mis_i
        mb_jac_mjs[i] = mb_jac_mjs_i
        dedcs[i] = dedcs_i
        dgdcs[i] = dgdcs_i
        dgmidcs[i] = dgmidcs_i
        dgmjdcs[i] = dgmjdcs_i

    for i in range(itypes.size):
        js_i = js[i, :]
        rs_i = rs[i, :, :]
        jtypes_i = jtypes[i, :]
        r_abs = _nb_linalg_norm(rs_i)
        r_unit = _calc_r_unit(rs_i, r_abs)
        store_mag_radial_basis(
            i,
            itypes[i],
            js_i,
            jtypes_i,
            rs_i,
            r_unit,
            rb_vals[i],
            drb_drs[i],
            drb_dmis[i],
            drb_dmjs[i],
            rbd_values,
            rbd_dqdris,
            rbd_dqdmis,
            rbd_dqdeps,
        )
        update_mbd_vatoms(i, mbd_vatoms, mb_vals[i])
        update_mbd_dbdris(i, js_i, mbd_dbdris, mb_jac_rs[i])
        update_mbd_dbdmis(i, js_i, mbd_dbdmis, mb_jac_mis[i], mb_jac_mjs[i])
        update_mbd_dbdeps(js_i, rs_i, mbd_dbdeps, mb_jac_rs[i])
        update_mbd_dvdcs(i, itypes[i], mbd_dedcs, dedcs[i])
        update_mbd_dgdcs(i, itypes[i], js_i, mbd_dgdcs, dgdcs[i])
        update_mbd_dgmdcs(i, itypes[i], js_i, mbd_dgmdcs, dgmidcs[i], dgmjdcs[i])
        update_mbd_dsdcs(itypes[i], js_i, rs_i, mbd_dsdcs, dgdcs[i])

    return energies
