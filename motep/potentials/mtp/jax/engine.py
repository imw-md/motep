from typing import Any

import jax.numpy as jnp
import numpy as np
from ase import Atoms

from motep.potentials.mtp.base import EngineBase
from motep.potentials.mtp.data import MTPData

from .conversion import BasisConverter
from .jax import calc_energy_forces_stress as jax_calc
from .moment import (
    extract_basic_moments,
    extract_pair_contractions,
    get_moment_contractions,
)

# dict mapping MLIP moments count to level, used for conversion
moments_count_to_level_map = {
    1: 2,
    2: 4,
    8: 6,
    18: 8,
    41: 10,
    84: 12,
    174: 14,
    350: 16,
    718: 18,
    1352: 20,
    2621: 22,
    4991: 24,
    9396: 26,
    17366: 28,
}


class JaxMTPEngine(EngineBase):
    """MTP Engine in 'full tensor' version based on jax."""

    def __init__(self, *args, **kwargs):
        """Intialize the engine."""
        self.scalar_contractions = None
        self.pair_contractions = None
        self.basic_moments = None
        self.basis_converter = BasisConverter(self)
        super().__init__(*args, **kwargs)

    def update(self, mtp_data: MTPData) -> None:
        """Update MTP parameters."""
        super().update(mtp_data)
        if self.mtp_data.alpha_moments_count is not None:
            if self.scalar_contractions is None:
                level = moments_count_to_level_map[mtp_data.alpha_moments_count]
                self.init_moment_mappings(level)
                self.basis_converter.remap_mlip_moment_coeffs()
            else:
                raise RuntimeError(
                    "Changing moments/level is not allowed. "
                    "Use a new instance instead."
                )

    def init_moment_mappings(self, level):
        self.scalar_contractions = get_moment_contractions(level)
        self.basic_moments = extract_basic_moments(self.scalar_contractions)
        self.pair_contractions = extract_pair_contractions(self.scalar_contractions)

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
            self.basic_moments,
            self.pair_contractions,
            self.scalar_contractions,
        )
        self.results["energy"] = energy
        self.results["forces"] = forces
        self.results["stress"] = stress
        return self.results["energy"], self.results["forces"], self.results["stress"]
