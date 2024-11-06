"""MTP written using numba."""

import numpy as np
from ase import Atoms

from motep.potentials.mtp.base import EngineBase

from .utils import (
    _nb_calc_local_energy_and_gradient,
    _nb_calc_moment,
    _nb_calc_radial_funcs,
    _nb_forces_from_gradient,
    _nb_linalg_norm,
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

        number_of_atoms = len(atoms)
        max_number_of_js, all_js, all_r_ijs = self._calc_max_ijs(atoms)
        itypes = [mtp_data["species"][atoms.numbers[i]] for i in range(number_of_atoms)]

        energy = 0.0
        stress = np.zeros((3, 3))
        gradient = np.zeros((number_of_atoms, max_number_of_js, 3))
        for i, itype in enumerate(itypes):
            js = all_js[:, i]
            r_ijs = all_r_ijs[i]
            (number_of_js, _) = r_ijs.shape
            jtypes = np.array([self.mtp_data["species"][atoms.numbers[j]] for j in js])
            r_abs = _nb_linalg_norm(r_ijs)
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
                r_ijs,
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
            stress = (stress + stress.T) * 0.5  # symmetrize
            stress /= atoms.get_volume()
            stress = stress.flat[[0, 4, 8, 5, 2, 1]]
        else:
            stress = np.full(6, np.nan)

        return energy, forces, stress

    def _calc_train(self, atoms: Atoms) -> tuple:
        self.update_neighbor_list(atoms)
        number_of_atoms = len(atoms)

        mtp_data = self.mtp_data

        itypes = [mtp_data["species"][atoms.numbers[i]] for i in range(number_of_atoms)]
        energies = self.mtp_data["species_coeffs"][itypes]

        self.mbd.clean()
        self.rbd.clean()

        moment_coeffs = mtp_data["moment_coeffs"]

        stress = np.zeros((3, 3))
        for i, itype in enumerate(itypes):
            js, r_ijs = self._get_distances(atoms, i)
            jtypes = np.array([self.mtp_data["species"][atoms.numbers[j]] for j in js])
            r_abs = _nb_linalg_norm(r_ijs)
            rb_values, rb_derivs = _nb_calc_radial_funcs(
                r_abs,
                itype,
                jtypes,
                mtp_data["radial_coeffs"],
                mtp_data["scaling"],
                mtp_data["min_dist"],
                mtp_data["max_dist"],
            )
            basis_values, basis_jac_rs = _nb_calc_moment(
                r_abs,
                r_ijs,
                rb_values,
                rb_derivs,
                mtp_data["alpha_moments_count"],
                mtp_data["alpha_moment_mapping"],
                mtp_data["alpha_index_basic"],
                mtp_data["alpha_index_times"],
            )

            self.mbd.values += basis_values

            energies[i] += moment_coeffs @ basis_values

            for k, j in enumerate(js):
                self.mbd.dbdris[:, i] -= basis_jac_rs[:, k]
                self.mbd.dbdris[:, j] += basis_jac_rs[:, k]
            self.mbd.dbdeps += r_ijs.T @ basis_jac_rs

        energy = energies.sum()
        forces = np.sum(moment_coeffs * self.mbd.dbdris.T, axis=-1).T * -1.0
        stress = np.sum(moment_coeffs * self.mbd.dbdeps.T, axis=-1).T

        self._symmetrize_stress(atoms, stress)

        stress = stress.flat[[0, 4, 8, 5, 2, 1]]

        return energy, forces, stress
