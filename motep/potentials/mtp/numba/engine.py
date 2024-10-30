"""MTP written using numba."""

from ase import Atoms

from motep.potentials.mtp.base import EngineBase


class NumbaMTPEngine(EngineBase):
    """MTP Engine based on Numba."""

    def calculate(self, atoms: Atoms):
        self.update_neighbor_list(atoms)
        return self.numba_calc_energy_and_forces(atoms)

    def numba_calc_energy_and_forces(self, atoms: Atoms):
        from .utils import numba_calc_energy_and_forces

        mlip_params = self.dict_mtp
        energy, forces, stress = numba_calc_energy_and_forces(
            self,
            atoms,
            mlip_params["alpha_moments_count"],
            mlip_params["alpha_moment_mapping"],
            mlip_params["alpha_index_basic"],
            mlip_params["alpha_index_times"],
            mlip_params["scaling"],
            mlip_params["min_dist"],
            mlip_params["max_dist"],
            mlip_params["species_coeffs"],
            mlip_params["moment_coeffs"],
            mlip_params["radial_coeffs"],
        )
        return energy, forces, stress
