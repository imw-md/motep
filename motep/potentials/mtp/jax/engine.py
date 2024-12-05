from typing import Any

import jax.numpy as jnp
import numpy as np
from ase import Atoms

from motep.potentials.mtp.base import EngineBase
from motep.potentials.mtp.data import MTPData

from .conversion import BasisConverter, moments_count_to_level_map
from .jax import calc_energy_forces_stress as jax_calc
from .moment import MomentBasis


class JaxMTPEngine(EngineBase):
    """MTP Engine in 'full tensor' version based on jax."""

    def __init__(self, *args, **kwargs):
        """Intialize the engine."""
        self.moment_basis = None
        self.basis_converter = None
        super().__init__(*args, **kwargs)

    def update(self, mtp_data: MTPData) -> None:
        """Update MTP parameters."""
        super().update(mtp_data)
        if self.mtp_data.alpha_moments_count is not None:
            level = moments_count_to_level_map[mtp_data.alpha_moments_count]
            if self.moment_basis is None:
                self.moment_basis = MomentBasis(level)
                self.moment_basis.init_moment_mappings()
                self.basis_converter = BasisConverter(self.moment_basis)
            elif self.moment_basis.max_level != level:
                raise RuntimeError(
                    "Changing moments/level is not allowed. "
                    "Use a new instance instead."
                )
            self.basis_converter.remap_mlip_moment_coeffs(self.mtp_data)

    def calculate(self, atoms: Atoms):
        self.update_neighbor_list(atoms)
        return self._calc_energy_forces_stress(atoms)

    def _get_all_distances(self, atoms: Atoms) -> tuple[np.ndarray, np.ndarray]:
        max_dist = self.mtp_data.max_dist
        max_num_js = np.max([_.shape[0] for _ in self.precomputed_offsets])
        all_js = []
        all_r_ijs = []
        for i in range(len(atoms)):
            js, r_ijs = self._get_distances(atoms, i)
            (num_js,) = js.shape
            pad = (0, max_num_js - num_js)
            padded_js = np.pad(js, pad_width=pad, constant_values=i)
            padded_rs = np.pad(r_ijs, pad_width=(pad, (0, 0)), constant_values=max_dist)
            all_js.append(padded_js)
            all_r_ijs.append(padded_rs)
        return jnp.array(all_js, dtype=int), jnp.array(all_r_ijs)

    def _calc_energy_forces_stress(self, atoms: Atoms):
        mtp_data = self.mtp_data
        energy, forces, stress = jax_calc(
            self,
            atoms,
            mtp_data.species,
            mtp_data.scaling,
            mtp_data.min_dist,
            mtp_data.max_dist,
            mtp_data.species_coeffs,
            self.basis_converter.remapped_coeffs,
            mtp_data.radial_coeffs,
            # Static parameters:
            self.moment_basis.basic_moments,
            self.moment_basis.pair_contractions,
            self.moment_basis.scalar_contractions,
        )
        self.results["energy"] = energy
        self.results["forces"] = forces
        self.results["stress"] = stress
        return self.results["energy"], self.results["forces"], self.results["stress"]
