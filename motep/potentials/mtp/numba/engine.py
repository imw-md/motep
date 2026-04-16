"""MTP written using numba."""

import numba as nb
import numpy as np
import numpy.typing as npt
from ase import Atoms

from motep.potentials.mtp.base import EngineBase
from motep.potentials.mtp.data import get_types

from .chebyshev import calc_radial_basis, sum_radial_terms
from .moment import (
    calc_moments_run,
    calc_moments_train,
    store_radial_basis,
    update_mbd_dbdeps,
    update_mbd_dbdris,
    update_mbd_dgdcs,
    update_mbd_dsdcs,
    update_mbd_dvdcs,
    update_mbd_vatoms,
)


class NumbaMTPEngine(EngineBase):
    """MTP Engine based on Numba."""

    def __init__(self, *args: tuple, **kwargs: dict) -> None:
        """Intialize the engine."""
        super().__init__(*args, **kwargs)

    def _calculate(self, atoms: Atoms) -> tuple:
        if self.mode == "run":
            return self._calc_run(atoms)
        if self.mode == "train":
            return self._calc_train(atoms)
        raise NotImplementedError(self.mode)

    def _calc_run(self, atoms: Atoms) -> tuple:
        mtp_data = self.mtp_data

        js = self._neighbors
        rs = self._get_interatomic_vectors(atoms)

        itypes = get_types(atoms, mtp_data.species)
        jtypes = itypes[js]

        self.mbd.clean()
        self.rbd.clean()

        energies, gradient = _calc_run(
            rs,
            itypes,
            jtypes,
            mtp_data.scaling,
            mtp_data.radial_basis.min,
            mtp_data.radial_basis.max,
            mtp_data.alpha_moments_count,
            mtp_data.alpha_index_basic,
            mtp_data.alpha_index_times,
            mtp_data.alpha_moment_mapping,
            mtp_data.radial_coeffs,
            mtp_data.species_coeffs,
            mtp_data.moment_coeffs,
            self.mbd.vatoms,
        )

        forces = _calc_forces_from_gradient(gradient, js)
        stress = np.einsum("ijk, ijl -> lk", rs, gradient)

        return energies, forces, stress

    def _calc_train(self, atoms: Atoms) -> tuple:
        mtp_data = self.mtp_data

        js = self._neighbors
        rs = self._get_interatomic_vectors(atoms)

        itypes = get_types(atoms, mtp_data.species)
        jtypes = itypes[js]

        self.mbd.clean()
        self.rbd.clean()

        energies = _calc_train(
            js,
            rs,
            itypes,
            jtypes,
            mtp_data.scaling,
            mtp_data.radial_basis.min,
            mtp_data.radial_basis.max,
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
            self.mbd.dvdcs,
            self.mbd.dgdcs,
            self.mbd.dsdcs,
        )

        moment_coeffs = mtp_data.moment_coeffs

        forces = np.sum(moment_coeffs * self.mbd.dbdris.T, axis=-1).T * -1.0
        stress = np.sum(moment_coeffs * self.mbd.dbdeps.T, axis=-1).T

        return energies, forces, stress


@nb.njit(nb.float64[:](nb.float64[:, :]))
def _nb_linalg_norm(r_ijs: np.ndarray) -> np.ndarray:
    r_abs = np.zeros((r_ijs.shape[0],))
    for j in range(r_ijs.shape[0]):
        for k in range(3):
            r_abs[j] += r_ijs[j, k] ** 2
        r_abs[j] **= 0.5
    return r_abs


@nb.njit(nb.float64[:, :](nb.float64[:, :], nb.float64[:]))
def _calc_r_unit(r_ijs: np.ndarray, r_abs: np.ndarray) -> np.ndarray:
    r_unit = np.zeros((r_ijs.shape[0], 3))
    for j in range(r_ijs.shape[0]):
        for k in range(3):
            r_unit[j, k] = r_ijs[j, k] / r_abs[j]
    return r_unit


@nb.njit(
    nb.types.Tuple((nb.float64[:], nb.float64[:, :, :]))(
        nb.float64[:, :, :],
        nb.int32[:],
        nb.int32[:, :],
        nb.float64,
        nb.float64,
        nb.float64,
        nb.int32,
        nb.int32[:, :],
        nb.int32[:, :],
        nb.int32[:],
        nb.float64[:, :, :, :],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:, :],  # mbd_vatoms
    ),
    # parallel=True,
)
def _calc_run(
    rs: npt.NDArray[np.float64],
    itypes: npt.NDArray[np.int32],
    jtypes: npt.NDArray[np.int32],
    scaling: np.float64,
    min_dist: np.float64,
    max_dist: np.float64,
    alpha_moments_count: np.int32,
    alpha_index_basic: npt.NDArray[np.int32],
    alpha_index_times: npt.NDArray[np.int32],
    alpha_moment_mapping: npt.NDArray[np.int32],
    radial_coeffs: npt.NDArray[np.float64],
    species_coeffs: npt.NDArray[np.float64],
    moment_coeffs: npt.NDArray[np.float64],
    mbd_vatoms: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:

    energies = species_coeffs[itypes]
    gradient = np.zeros((itypes.size, jtypes.shape[1], 3))
    mb_vals = np.empty((itypes.size, alpha_moment_mapping.size))

    for i in nb.prange(itypes.size):
        r_abs = _nb_linalg_norm(rs[i, :, :])
        r_unit = _calc_r_unit(rs[i, :, :], r_abs)

        rbs = radial_coeffs.shape[3]
        rbasis, drbdrs = calc_radial_basis(r_abs, rbs, scaling, min_dist, max_dist)
        rfvals = sum_radial_terms(itypes[i], jtypes[i, :], rbasis, radial_coeffs)
        drfdrs = sum_radial_terms(itypes[i], jtypes[i, :], drbdrs, radial_coeffs)

        mb_values, local_gradient = calc_moments_run(
            r_unit,
            r_abs,
            rfvals,
            drfdrs,
            alpha_moments_count,
            alpha_moment_mapping,
            alpha_index_basic,
            alpha_index_times,
            moment_coeffs,
        )

        mb_vals[i] = mb_values

        for basis_i, coeff in enumerate(moment_coeffs):
            energies[i] += coeff * mb_values[basis_i]

        for j in range(r_abs.size):
            for k in range(3):
                gradient[i, j, k] = local_gradient[j, k]

    for i in range(itypes.size):
        update_mbd_vatoms(i, mbd_vatoms, mb_vals[i])

    return energies, gradient


@nb.njit(
    nb.float64[:](
        nb.int32[:, :],
        nb.float64[:, :, :],
        nb.int32[:],
        nb.int32[:, :],
        nb.float64,
        nb.float64,
        nb.float64,
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
        nb.float64[:, :],  # mbd_vatoms
        nb.float64[:, :, :],
        nb.float64[:, :, :],
        nb.float64[:, :, :, :, :],  # mbd_dvdcs
        nb.float64[:, :, :, :, :, :],
        nb.float64[:, :, :, :, :, :],
    ),
    # parallel=True,
)
def _calc_train(
    js: npt.NDArray[np.int32],
    rs: npt.NDArray[np.float64],
    itypes: npt.NDArray[np.int32],
    jtypes: npt.NDArray[np.int32],
    scaling: np.float64,
    min_dist: np.float64,
    max_dist: np.float64,
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
    mbd_dvdcs: npt.NDArray[np.float64],
    mbd_dgdcs: npt.NDArray[np.float64],
    mbd_dsdcs: npt.NDArray[np.float64],
):
    _, species_count, rfs, rbs = radial_coeffs.shape
    rb_vals = np.empty((itypes.size, radial_coeffs.shape[3], js.shape[1]))
    rb_ders = np.empty((itypes.size, radial_coeffs.shape[3], js.shape[1]))
    mb_vals = np.empty((itypes.size, alpha_moment_mapping.size))
    mb_ders = np.empty((itypes.size, alpha_moment_mapping.size, *rs.shape[1:]))
    dvdcs_l = np.empty((itypes.size, species_count, rfs, rbs))
    dgdcs_l = np.empty((itypes.size, species_count, rfs, rbs, *rs.shape[1:]))

    energies = species_coeffs[itypes]
    for i in nb.prange(itypes.size):
        js_i = js[i, :]
        rs_i = rs[i, :, :]
        jtypes_i = jtypes[i, :]
        r_abs = _nb_linalg_norm(rs_i)
        r_unit = _calc_r_unit(rs_i, r_abs)
        rb_values, rb_derivs = calc_radial_basis(
            r_abs,
            radial_coeffs.shape[3],
            scaling,
            min_dist,
            max_dist,
        )
        basis_values, basis_jac_rs, dvdcs, dgdcs = calc_moments_train(
            itypes[i],
            jtypes_i,
            r_abs,
            r_unit,
            rb_values,
            rb_derivs,
            radial_coeffs,
            alpha_moments_count,
            alpha_moment_mapping,
            alpha_index_basic,
            alpha_index_times,
            moment_coeffs,
        )

        for basis_i, coeff in enumerate(moment_coeffs):
            energies[i] += coeff * basis_values[basis_i]

        rb_vals[i] = rb_values
        rb_ders[i] = rb_derivs
        mb_vals[i] = basis_values
        mb_ders[i] = basis_jac_rs
        dvdcs_l[i] = dvdcs
        dgdcs_l[i] = dgdcs

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
            rb_ders[i],
            rbd_values,
            rbd_dqdris,
            rbd_dqdeps,
        )
        update_mbd_vatoms(i, mbd_vatoms, mb_vals[i])
        update_mbd_dbdris(i, js_i, mbd_dbdris, mb_ders[i])
        update_mbd_dbdeps(js_i, rs_i, mbd_dbdeps, mb_ders[i])
        update_mbd_dvdcs(i, itypes[i], mbd_dvdcs, dvdcs_l[i])
        update_mbd_dgdcs(i, itypes[i], js_i, mbd_dgdcs, dgdcs_l[i])
        update_mbd_dsdcs(itypes[i], js_i, rs_i, mbd_dsdcs, dgdcs_l[i])

    return energies


@nb.njit(
    nb.float64[:, :](
        nb.float64[:, :, :],
        nb.int32[:, :],
    ),
)
def _calc_forces_from_gradient(
    gradient: np.ndarray,
    all_js: np.ndarray,
) -> np.ndarray:
    forces = np.zeros((gradient.shape[0], 3))
    for i in range(gradient.shape[0]):
        for i_j in range(gradient.shape[1]):
            j = all_js[i, i_j]
            for k in range(3):
                forces[i, k] += gradient[i, i_j, k]
                forces[j, k] -= gradient[i, i_j, k]
    return forces
