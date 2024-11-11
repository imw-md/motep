"""MTP written using numba."""

import numba as nb
import numpy as np
import numpy.typing as npt
from ase import Atoms

from motep.potentials.mtp import get_types
from motep.potentials.mtp.base import EngineBase

from .chebyshev import _nb_calc_radial_basis, _nb_calc_radial_funcs
from .utils import (
    _nb_calc_local_energy_and_gradient,
    _nb_calc_moment,
    _nb_forces_from_gradient,
    _store_radial_basis_values,
)


class NumbaMTPEngine(EngineBase):
    """MTP Engine based on Numba."""

    def __init__(self, *args: tuple, **kwargs: dict) -> None:
        """Intialize the engine."""
        super().__init__(*args, **kwargs)
        self.calculate = self._calc_train if self._is_trained else self._calc_run

    def _calc_max_ijs(self, atoms: Atoms) -> tuple:
        # TODO: precompute distances and send in indices.
        # See also jax implementation of full tensor version
        number_of_atoms = len(atoms)
        max_number_of_js = 0
        all_js_list = []
        all_r_ijs = []
        for i in range(number_of_atoms):
            js, r_ijs = self._get_distances(atoms, i)
            all_js_list.append(js)
            all_r_ijs.append(r_ijs)
            (number_of_js,) = js.shape
            max_number_of_js = max(number_of_js, max_number_of_js)
        shape = (max_number_of_js, number_of_atoms)
        all_js = np.zeros(shape, dtype=int)
        for i in range(number_of_atoms):
            js = all_js_list[i]
            (number_of_js,) = js.shape
            all_js[:number_of_js, i] = js
        return max_number_of_js, all_js, all_r_ijs

    def _calc_run(self, atoms: Atoms) -> tuple:
        """Calculate properties of the given system."""
        self.update_neighbor_list(atoms)

        mtp_data = self.mtp_data

        max_number_of_js, all_js, all_r_ijs = self._calc_max_ijs(atoms)
        itypes = get_types(atoms, self.mtp_data["species"])

        energy = 0.0
        stress = np.zeros((3, 3))
        gradient = np.zeros((itypes.size, max_number_of_js, 3))
        for i, itype in enumerate(itypes):
            js = all_js[:, i]
            r_ijs = all_r_ijs[i]
            (number_of_js, _) = r_ijs.shape
            jtypes = itypes[js]
            r_abs = _nb_linalg_norm(r_ijs)
            r_ijs_unit = _calc_r_unit(r_ijs, r_abs)
            rb_values, rb_derivs = _nb_calc_radial_funcs(
                r_abs,
                itype,
                jtypes,
                mtp_data["radial_coeffs"],
                mtp_data["scaling"],
                mtp_data["min_dist"],
                mtp_data["max_dist"],
            )
            local_energy, local_gradient = _nb_calc_local_energy_and_gradient(
                r_ijs_unit,
                r_abs,
                rb_values,
                rb_derivs,
                mtp_data["alpha_moments_count"],
                mtp_data["alpha_moment_mapping"],
                mtp_data["alpha_index_basic"],
                mtp_data["alpha_index_times"],
                itype,
                mtp_data["species_coeffs"],
                mtp_data["moment_coeffs"],
            )
            energy += local_energy
            stress += r_ijs.T @ local_gradient
            gradient[i, :number_of_js, :] = local_gradient

        forces = _nb_forces_from_gradient(gradient, all_js, max_number_of_js)

        if atoms.cell.rank == 3:
            stress += stress.T  # symmetrize
            stress *= 0.5 / atoms.get_volume()
        else:
            stress[:, :] = np.nan

        stress = stress.flat[[0, 4, 8, 5, 2, 1]]

        return energy, forces, stress

    def _calc_train(self, atoms: Atoms) -> tuple:
        self.update_neighbor_list(atoms)

        mtp_data = self.mtp_data

        itypes = get_types(atoms, self.mtp_data["species"])
        energies = self.mtp_data["species_coeffs"][itypes]

        self.mbd.clean()
        self.rbd.clean()

        moment_coeffs = mtp_data["moment_coeffs"]

        stress = np.zeros((3, 3))
        for i, itype in enumerate(itypes):
            js, r_ijs = self._get_distances(atoms, i)
            jtypes = itypes[js]
            r_abs = _nb_linalg_norm(r_ijs)
            r_ijs_unit = _calc_r_unit(r_ijs, r_abs)
            rb_values, rb_derivs = _nb_calc_radial_basis(
                r_abs,
                mtp_data["radial_basis_size"],
                mtp_data["scaling"],
                mtp_data["min_dist"],
                mtp_data["max_dist"],
            )
            _store_radial_basis_values(
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
            basis_values, basis_jac_rs, dedcs, dgdcs = _nb_calc_moment(
                itype,
                jtypes,
                r_abs,
                r_ijs_unit,
                rb_values,
                rb_derivs,
                mtp_data["radial_coeffs"],
                mtp_data["alpha_moments_count"],
                mtp_data["alpha_moment_mapping"],
                mtp_data["alpha_index_basic"],
                mtp_data["alpha_index_times"],
                mtp_data["moment_coeffs"],
            )

            energies[i] += moment_coeffs @ basis_values

            _update_moment_basis_data_props(
                i,
                js,
                r_ijs,
                self.mbd.values,
                self.mbd.dbdris,
                self.mbd.dbdeps,
                basis_values,
                basis_jac_rs,
            )

            _update_moment_basis_data_dcs(
                i,
                itype,
                js,
                r_ijs,
                self.mbd.dedcs,
                self.mbd.dgdcs,
                self.mbd.dsdcs,
                dedcs,
                dgdcs,
            )

        energy = energies.sum()
        forces = np.sum(moment_coeffs * self.mbd.dbdris.T, axis=-1).T * -1.0
        stress = np.sum(moment_coeffs * self.mbd.dbdeps.T, axis=-1).T

        self._symmetrize_stress(atoms, stress)

        stress = stress.flat[[0, 4, 8, 5, 2, 1]]

        return energy, forces, stress


@nb.njit(nb.float64[:, :](nb.float64[:, :], nb.float64[:]))
def _calc_r_unit(r_ijs: np.ndarray, r_abs: np.ndarray) -> np.ndarray:
    return r_ijs[:, :] / r_abs[:, None]


@nb.njit(nb.float64[:](nb.float64[:, :]))
def _nb_linalg_norm(r_ijs: np.ndarray) -> np.ndarray:
    return np.sqrt((r_ijs**2).sum(axis=1))


@nb.njit
def _update_moment_basis_data_props(
    i: np.int64,
    js: npt.NDArray[np.int64],
    r_ijs: npt.NDArray[np.float64],
    mbd_values: npt.NDArray[np.float64],
    mbd_dbdris: npt.NDArray[np.float64],
    mbd_dbdeps: npt.NDArray[np.float64],
    basis_values: npt.NDArray[np.float64],
    basis_jac_rs: npt.NDArray[np.float64],
):
    """Update `MomentBasisData` energies, gradients, and stresses."""
    mbd_values += basis_values
    for k, j in enumerate(js):
        mbd_dbdris[:, i] -= basis_jac_rs[:, k]
        mbd_dbdris[:, j] += basis_jac_rs[:, k]
        for ixyz0 in range(3):
            for ixyz1 in range(3):
                mbd_dbdeps[:, ixyz0, ixyz1] += (
                    r_ijs[k, ixyz0] * basis_jac_rs[:, k, ixyz1]
                )


@nb.njit
def _update_moment_basis_data_dcs(
    i: np.int64,
    itype: np.int64,
    js: npt.NDArray[np.int64],
    r_ijs: npt.NDArray[np.float64],
    mbd_dedcs: npt.NDArray[np.float64],
    mbd_dgdcs: npt.NDArray[np.float64],
    mbd_dsdcs: npt.NDArray[np.float64],
    tmp_dedcs: npt.NDArray[np.float64],
    tmp_dgdcs: npt.NDArray[np.float64],
) -> None:
    """Update `MomentBasisData` Jacobians with respect to radial coefficients."""
    mbd_dedcs[itype] += tmp_dedcs
    for k, j in enumerate(js):
        mbd_dgdcs[itype, :, :, :, i] -= tmp_dgdcs[:, :, :, k]
        mbd_dgdcs[itype, :, :, :, j] += tmp_dgdcs[:, :, :, k]
        for ixyz0 in range(3):
            for ixyz1 in range(3):
                mbd_dsdcs[itype, :, :, :, ixyz0, ixyz1] += (
                    r_ijs[k, ixyz0] * tmp_dgdcs[:, :, :, k, ixyz1]
                )
