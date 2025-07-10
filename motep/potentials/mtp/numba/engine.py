"""MTP written using numba."""

import numba as nb
import numpy as np
import numpy.typing as npt
from ase import Atoms

from motep.potentials.mtp import get_types
from motep.potentials.mtp.base import EngineBase

from .chebyshev import nb_calc_radial_basis, nb_calc_radial_funcs
from .moment import (
    nb_calc_local_energy_and_gradient,
    nb_calc_moment,
    store_radial_basis_values,
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
        self.calculate = self._calc_train if self._is_trained else self._calc_run

    def _calc_run(self, atoms: Atoms) -> tuple:
        """Calculate properties of the given system."""
        self.update_neighbor_list(atoms)

        mtp_data = self.mtp_data

        all_js, all_r_ijs = self._get_all_distances(atoms)
        itypes = get_types(atoms, self.mtp_data.species)
        all_jtypes = itypes[all_js]
        energies, gradient = _calc_energy_and_gradient(
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
        )

        forces = _calc_forces_from_gradient(gradient, all_js)
        stress = np.einsum("ijk, ijl -> lk", all_r_ijs, gradient)

        self._symmetrize_stress(atoms, stress)

        self.results["energies"] = energies
        self.results["energy"] = self.results["energies"].sum()
        self.results["forces"] = forces
        self.results["stress"] = stress.flat[[0, 4, 8, 5, 2, 1]]

        return self.results

    def _calc_train(self, atoms: Atoms) -> tuple:
        self.update_neighbor_list(atoms)

        mtp_data = self.mtp_data

        itypes = get_types(atoms, self.mtp_data.species)
        energies = self.mtp_data.species_coeffs[itypes]

        self.mbd.clean()
        self.rbd.clean()

        moment_coeffs = mtp_data.moment_coeffs

        stress = np.zeros((3, 3))
        for i, itype in enumerate(itypes):
            js, r_ijs = self._get_distances(atoms, i)
            jtypes = itypes[js]
            r_abs = _nb_linalg_norm(r_ijs)
            r_ijs_unit = _calc_r_unit(r_ijs, r_abs)
            rb_values, rb_derivs = nb_calc_radial_basis(
                r_abs,
                mtp_data.radial_basis_size,
                mtp_data.scaling,
                mtp_data.min_dist,
                mtp_data.max_dist,
            )
            store_radial_basis_values(
                i,
                itype,
                js,
                jtypes,
                r_ijs,
                r_ijs_unit,
                rb_values,
                rb_derivs,
                self.rbd.values,
                self.rbd.dqdris,
                self.rbd.dqdeps,
            )
            basis_values, basis_jac_rs, dedcs, dgdcs = nb_calc_moment(
                itype,
                jtypes,
                r_abs,
                r_ijs_unit,
                rb_values,
                rb_derivs,
                mtp_data.radial_coeffs,
                mtp_data.alpha_moments_count,
                mtp_data.alpha_moment_mapping,
                mtp_data.alpha_index_basic,
                mtp_data.alpha_index_times,
                mtp_data.moment_coeffs,
            )

            energies[i] += moment_coeffs @ basis_values

            update_mbd_values(self.mbd.values, basis_values)
            update_mbd_dbdris(i, js, self.mbd.dbdris, basis_jac_rs)
            update_mbd_dbdeps(js, r_ijs, self.mbd.dbdeps, basis_jac_rs)
            update_mbd_dedcs(itype, self.mbd.dedcs, dedcs)
            update_mbd_dgdcs(i, itype, js, self.mbd.dgdcs, dgdcs)
            update_mbd_dsdcs(itype, js, r_ijs, self.mbd.dsdcs, dgdcs)

        forces = np.sum(moment_coeffs * self.mbd.dbdris.T, axis=-1).T * -1.0
        stress = np.sum(moment_coeffs * self.mbd.dbdeps.T, axis=-1).T

        self._symmetrize_stress(atoms, stress)

        stress = stress.flat[[0, 4, 8, 5, 2, 1]]

        self.results["energies"] = energies
        self.results["energy"] = self.results["energies"].sum()
        self.results["forces"] = forces
        self.results["stress"] = stress

        return self.results


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
    ),
    # parallel=True,
)
def _calc_energy_and_gradient(
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
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    energies = species_coeffs[itypes]
    gradient = np.zeros((itypes.size, all_jtypes.shape[1], 3))
    for i in nb.prange(itypes.size):
        r_abs = _nb_linalg_norm(all_r_ijs[i, :, :])
        r_ijs_unit = _calc_r_unit(all_r_ijs[i, :, :], r_abs)
        rb_values, rb_derivs = nb_calc_radial_funcs(
            r_abs,
            itypes[i],
            all_jtypes[i, :],
            radial_coeffs,
            scaling,
            min_dist,
            max_dist,
        )
        local_energy, local_gradient = nb_calc_local_energy_and_gradient(
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
        energies[i] += local_energy
        for j in range(all_jtypes.shape[1]):
            for k in range(3):
                gradient[i, j, k] = local_gradient[j, k]
    return energies, gradient


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
