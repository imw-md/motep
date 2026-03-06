"""MTP written using numba."""

import numba as nb
import numpy as np
import numpy.typing as npt
from ase import Atoms

from motep.potentials.mtp import get_types
from motep.potentials.mtp.base import EngineBase

from .chebyshev import calc_radial_basis, calc_radial_funcs
from .moment import (
    calc_moments_run,
    calc_moments_train,
    store_radial_basis,
    update_mbd_dbdeps,
    update_mbd_dbdris,
    update_mbd_dedcs,
    update_mbd_dgdcs,
    update_mbd_dsdcs,
    update_mbd_values,
)


class NumbaMTPEngine(EngineBase):
    """MTP Engine based on Numba."""

    def __init__(self, *args: tuple, **kwargs: dict) -> None:
        """Intialize the engine."""
        super().__init__(*args, **kwargs)

    def _calculate(self, atoms: Atoms) -> tuple:
        if self._is_trained:
            return self._calc_train(atoms)
        return self._calc_run(atoms)

    def _calc_run(self, atoms: Atoms) -> tuple:
        mtp_data = self.mtp_data

        all_js, all_r_ijs = self._get_all_distances(atoms)

        itypes = get_types(atoms, mtp_data.species)
        all_jtypes = itypes[all_js]

        self.mbd.clean()
        self.rbd.clean()

        energies, gradient = _calc_run(
            all_r_ijs,
            itypes,
            all_jtypes,
            mtp_data.alpha_moments_count,
            mtp_data.alpha_moment_mapping,
            mtp_data.alpha_index_basic,
            mtp_data.alpha_index_times,
            mtp_data.scaling,
            mtp_data.min_dist,
            mtp_data.max_dist,
            mtp_data.radial_coeffs,
            mtp_data.species_coeffs,
            mtp_data.moment_coeffs,
            self.mbd.values,
        )

        forces = _calc_forces_from_gradient(gradient, all_js)
        stress = np.einsum("ijk, ijl -> lk", all_r_ijs, gradient)

        return energies, forces, stress

    def _calc_train(self, atoms: Atoms) -> tuple:
        mtp_data = self.mtp_data

        all_js, all_r_ijs = self.all_js, self.all_r_ijs

        itypes = get_types(atoms, mtp_data.species)
        all_jtypes = itypes[all_js]

        self.mbd.clean()
        self.rbd.clean()

        energies = _calc_train(
            all_js,
            all_r_ijs,
            itypes,
            all_jtypes,
            mtp_data.alpha_moments_count,
            mtp_data.alpha_moment_mapping,
            mtp_data.alpha_index_basic,
            mtp_data.alpha_index_times,
            mtp_data.scaling,
            mtp_data.min_dist,
            mtp_data.max_dist,
            mtp_data.radial_coeffs,
            mtp_data.species_coeffs,
            mtp_data.moment_coeffs,
            self.rbd.values,
            self.rbd.dqdris,
            self.rbd.dqdeps,
            self.mbd.values,
            self.mbd.dbdris,
            self.mbd.dbdeps,
            self.mbd.dedcs,
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
    r_ijs_unit = np.zeros((r_ijs.shape[0], 3))
    for j in range(r_ijs.shape[0]):
        for k in range(3):
            r_ijs_unit[j, k] = r_ijs[j, k] / r_abs[j]
    return r_ijs_unit


@nb.njit(
    nb.types.Tuple((nb.float64[:], nb.float64[:, :, :]))(
        nb.float64[:, :, :],
        nb.int64[:],
        nb.int64[:, :],
        nb.int64,
        nb.int64[:],
        nb.int64[:, :],
        nb.int64[:, :],
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64[:, :, :, :],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:],
    ),
    parallel=True,
)
def _calc_run(
    all_r_ijs: npt.NDArray[np.float64],
    itypes: npt.NDArray[np.int64],
    all_jtypes: npt.NDArray[np.int64],
    alpha_moments_count: np.int64,
    alpha_moment_mapping: npt.NDArray[np.int64],
    alpha_index_basic: npt.NDArray[np.int64],
    alpha_index_times: npt.NDArray[np.int64],
    scaling: np.float64,
    min_dist: np.float64,
    max_dist: np.float64,
    radial_coeffs: npt.NDArray[np.float64],
    species_coeffs: npt.NDArray[np.float64],
    moment_coeffs: npt.NDArray[np.float64],
    mbd_values: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:

    energies = species_coeffs[itypes]
    gradient = np.zeros((itypes.size, all_jtypes.shape[1], 3))

    for i in nb.prange(itypes.size):
        r_abs = _nb_linalg_norm(all_r_ijs[i, :, :])
        r_ijs_unit = _calc_r_unit(all_r_ijs[i, :, :], r_abs)
        rb_values, rb_derivs = calc_radial_funcs(
            r_abs,
            itypes[i],
            all_jtypes[i, :],
            radial_coeffs,
            scaling,
            min_dist,
            max_dist,
        )
        basis_values, local_gradient = calc_moments_run(
            r_ijs_unit,
            r_abs,
            rb_values,
            rb_derivs,
            alpha_moments_count,
            alpha_moment_mapping,
            alpha_index_basic,
            alpha_index_times,
            moment_coeffs,
        )

        update_mbd_values(mbd_values, basis_values)

        for basis_i, coeff in enumerate(moment_coeffs):
            energies[i] += coeff * basis_values[basis_i]

        for j in range(r_abs.size):
            for k in range(3):
                gradient[i, j, k] = local_gradient[j, k]

    return energies, gradient


@nb.njit(
    nb.float64[:](
        nb.int64[:, :],
        nb.float64[:, :, :],
        nb.int64[:],
        nb.int64[:, :],
        nb.int64,
        nb.int64[:],
        nb.int64[:, :],
        nb.int64[:, :],
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64[:, :, :, :],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:, :, :],
        nb.float64[:, :, :, :, :],
        nb.float64[:, :, :, :, :],
        nb.float64[:],
        nb.float64[:, :, :],
        nb.float64[:, :, :],
        nb.float64[:, :, :, :],
        nb.float64[:, :, :, :, :, :],
        nb.float64[:, :, :, :, :, :],
    ),
    parallel=True,
)
def _calc_train(
    all_js: npt.NDArray[np.int64],
    all_r_ijs: npt.NDArray[np.float64],
    itypes: npt.NDArray[np.int64],
    all_jtypes: npt.NDArray[np.int64],
    alpha_moments_count: np.int64,
    alpha_moment_mapping: npt.NDArray[np.int64],
    alpha_index_basic: npt.NDArray[np.int64],
    alpha_index_times: npt.NDArray[np.int64],
    scaling: np.float64,
    min_dist: np.float64,
    max_dist: np.float64,
    radial_coeffs: npt.NDArray[np.float64],
    species_coeffs: npt.NDArray[np.float64],
    moment_coeffs: npt.NDArray[np.float64],
    rbd_values: npt.NDArray[np.float64],
    rbd_dqdris: npt.NDArray[np.float64],
    rbd_dqdeps: npt.NDArray[np.float64],
    mbd_values: npt.NDArray[np.float64],
    mbd_dbdris: npt.NDArray[np.float64],
    mbd_dbdeps: npt.NDArray[np.float64],
    mbd_dedcs: npt.NDArray[np.float64],
    mbd_dgdcs: npt.NDArray[np.float64],
    mbd_dsdcs: npt.NDArray[np.float64],
):
    _, species_count, rfs, rbs = radial_coeffs.shape
    rb_vals = np.empty((itypes.size, radial_coeffs.shape[3], all_js.shape[1]))
    rb_ders = np.empty((itypes.size, radial_coeffs.shape[3], all_js.shape[1]))
    mb_vals = np.empty((itypes.size, alpha_moment_mapping.size))
    mb_ders = np.empty((itypes.size, alpha_moment_mapping.size, *all_r_ijs.shape[1:]))
    dedcs_l = np.empty((itypes.size, species_count, rfs, rbs))
    dgdcs_l = np.empty((itypes.size, species_count, rfs, rbs, *all_r_ijs.shape[1:]))

    energies = species_coeffs[itypes]
    for i in nb.prange(itypes.size):
        js = all_js[i, :]
        r_ijs = all_r_ijs[i, :, :]
        jtypes = all_jtypes[i, :]
        r_abs = _nb_linalg_norm(r_ijs)
        r_ijs_unit = _calc_r_unit(r_ijs, r_abs)
        rb_values, rb_derivs = calc_radial_basis(
            r_abs,
            radial_coeffs.shape[3],
            scaling,
            min_dist,
            max_dist,
        )
        basis_values, basis_jac_rs, dedcs, dgdcs = calc_moments_train(
            itypes[i],
            jtypes,
            r_abs,
            r_ijs_unit,
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
        dedcs_l[i] = dedcs
        dgdcs_l[i] = dgdcs

    for i in range(itypes.size):
        js = all_js[i, :]
        r_ijs = all_r_ijs[i, :, :]
        jtypes = all_jtypes[i, :]
        r_abs = _nb_linalg_norm(r_ijs)
        r_ijs_unit = _calc_r_unit(r_ijs, r_abs)
        store_radial_basis(
            i,
            itypes[i],
            js,
            jtypes,
            r_ijs,
            r_ijs_unit,
            rb_vals[i],
            rb_ders[i],
            rbd_values,
            rbd_dqdris,
            rbd_dqdeps,
        )
        update_mbd_values(mbd_values, mb_vals[i])
        update_mbd_dbdris(i, js, mbd_dbdris, mb_ders[i])
        update_mbd_dbdeps(js, r_ijs, mbd_dbdeps, mb_ders[i])
        update_mbd_dedcs(itypes[i], mbd_dedcs, dedcs_l[i])
        update_mbd_dgdcs(i, itypes[i], js, mbd_dgdcs, dgdcs_l[i])
        update_mbd_dsdcs(itypes[i], js, r_ijs, mbd_dsdcs, dgdcs_l[i])

    return energies


@nb.njit(
    nb.float64[:, :](
        nb.float64[:, :, :],
        nb.int64[:, :],
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
