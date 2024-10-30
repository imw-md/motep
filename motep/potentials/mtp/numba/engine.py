"""MTP written using numba."""

import numpy as np
from ase import Atoms

from motep.potentials.mtp.base import EngineBase


class NumbaMTPEngine(EngineBase):
    """MTP Engine based on Numba."""

    def calculate(self, atoms: Atoms):
        """Calculate properties of the given system."""
        from .utils import (
            _nb_calc_local_energy_and_gradient,
            _nb_calc_radial_basis,
            _nb_forces_from_gradient,
        )

        self.update_neighbor_list(atoms)

        mtp_data = self.dict_mtp

        alpha_moments_count = mtp_data["alpha_moments_count"]
        alpha_moment_mapping = mtp_data["alpha_moment_mapping"]
        alpha_index_basic = mtp_data["alpha_index_basic"]
        alpha_index_times = mtp_data["alpha_index_times"]
        scaling = mtp_data["scaling"]
        min_dist = mtp_data["min_dist"]
        max_dist = mtp_data["max_dist"]
        species_coeffs = mtp_data["species_coeffs"]
        moment_coeffs = mtp_data["moment_coeffs"]
        radial_coeffs = mtp_data["radial_coeffs"]

        assert len(alpha_index_times.shape) == 2
        number_of_atoms = len(atoms)
        # TODO: precompute distances and send in indices.
        # See also jax implementation of full tensor version
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

        itypes = [mtp_data["species"][atoms.numbers[i]] for i in range(number_of_atoms)]

        energy = 0
        stress = np.zeros((3, 3))
        gradient = np.zeros((number_of_atoms, max_number_of_js, 3))
        for i, itype in enumerate(itypes):
            js = all_js[:, i]
            r_ijs = all_r_ijs[i]
            (number_of_js, _) = r_ijs.shape
            jtypes = np.array([self.dict_mtp["species"][atoms.numbers[j]] for j in js])
            r_abs = np.sqrt(np.add.reduce(r_ijs**2, axis=1))
            rb_values, rb_derivs = _nb_calc_radial_basis(
                r_abs, itype, jtypes, radial_coeffs, scaling, min_dist, max_dist
            )
            local_energy, local_gradient = _nb_calc_local_energy_and_gradient(
                r_ijs,
                r_abs,
                rb_values,
                rb_derivs,
                alpha_moments_count,
                alpha_moment_mapping,
                alpha_index_basic,
                alpha_index_times,
                itype,
                species_coeffs,
                moment_coeffs,
            )
            energy += local_energy
            stress += r_ijs.T @ local_gradient.T
            gradient[i, :number_of_js, :] = local_gradient.T

        forces = _nb_forces_from_gradient(
            gradient, all_js, number_of_atoms, max_number_of_js
        )

        if atoms.cell.rank == 3:
            stress = (stress + stress.T) * 0.5  # symmetrize
            stress /= atoms.get_volume()
            stress = stress.flat[[0, 4, 8, 5, 2, 1]]
        else:
            stress = np.full(6, np.nan)

        return energy, forces, stress
