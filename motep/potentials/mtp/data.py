"""Initializer."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass
class MTPData:
    """Subclass of `dict` to handle MTP parameters."""

    version: str = ""
    potential_name: str = ""
    scaling: float = 1.0
    species_count: int = 0
    potential_tag: str = ""
    radial_basis_type: str = ""
    min_dist: np.float64 = np.nan
    max_dist: np.float64 = np.nan
    radial_funcs_count: int = 0
    radial_basis_size: int = 0
    radial_coeffs: npt.NDArray[np.float64] | None = None
    alpha_moments_count: int = 0
    alpha_index_basic_count: int = 0
    alpha_index_basic: npt.NDArray[np.int64] | None = None
    alpha_index_times_count: int = 0
    alpha_index_times: npt.NDArray[np.int64] | None = None
    alpha_scalar_moments: int = 0
    alpha_moment_mapping: npt.NDArray[np.int64] | None = None
    species_coeffs: npt.NDArray[np.float64] | None = None
    moment_coeffs: npt.NDArray[np.float64] | None = None
    species: npt.NDArray[np.int64] | None = None
    optimized: list[str] = field(
        default_factory=lambda: ["species_coeffs", "moment_coeffs", "radial_coeffs"],
    )

    def initialize(self, rng: np.random.Generator) -> None:
        """Initialize MTP parameters.

        Parameters
        ----------
        rng : np.random.Generator
            Pseudo-random-number generator (PRNG) with the NumPy API.

        """
        if self.species_coeffs is None:
            self.species_coeffs = rng.uniform(-5.0, +5.0, self.species_count)
        if self.moment_coeffs is None:
            self.moment_coeffs = rng.uniform(-5.0, +5.0, self.alpha_scalar_moments)
        if self.radial_coeffs is None:
            spc = self.species_count
            rfc = self.radial_funcs_count
            rbs = self.radial_basis_size
            self.radial_coeffs = rng.uniform(-0.1, +0.1, (spc, spc, rfc, rbs))

    @property
    def parameters(self) -> np.ndarray:
        """Serialized parameters."""
        tmp = []
        if "scaling" in self.optimized:
            tmp.append(np.atleast_1d(self.scaling))
        if "moment_coeffs" in self.optimized:
            tmp.append(self.moment_coeffs)
        if "species_coeffs" in self.optimized:
            tmp.append(self.species_coeffs)
        if "radial_coeffs" in self.optimized:
            tmp.append(self.radial_coeffs.flat)
        return np.hstack(tmp)

    @parameters.setter
    def parameters(self, parameters: list[float]) -> None:
        """Update data in the .mtp file.

        Parameters
        ----------
        parameters : list[float]
            MTP parameters.

        """
        species_count = self.species_count
        rfc = self.radial_funcs_count
        rbs = self.radial_basis_size
        asm = self.alpha_scalar_moments

        n = 0
        if "scaling" in self.optimized:
            self.scaling = parameters[n]
            n += 1
        if "moment_coeffs" in self.optimized:
            self.moment_coeffs = parameters[n : asm + n]
            n += asm
        if "species_coeffs" in self.optimized:
            self.species_coeffs = parameters[n : n + species_count]
            n += species_count
        if "radial_coeffs" in self.optimized:
            total_radial = parameters[n:]
            shape = species_count, species_count, rfc, rbs
            self.radial_coeffs = np.array(total_radial).reshape(shape)

    def get_bounds(self) -> np.ndarray:
        """Get bounds."""
        tmp = []
        if "scaling" in self.optimized:
            tmp.append((0.0, np.inf))
        if "moment_coeffs" in self.optimized:
            tmp.extend([(-np.inf, +np.inf)] * self.moment_coeffs.size)
        if "species_coeffs" in self.optimized:
            tmp.extend([(-np.inf, +np.inf)] * self.species_coeffs.size)
        if "radial_coeffs" in self.optimized:
            tmp.extend([(-np.inf, +np.inf)] * self.radial_coeffs.size)
        return np.vstack(tmp)

    def print(self, **kwargs: dict[str, Any]) -> None:
        """Print parameters.

        `**kwargs` are used to, e.g., give `flush=True` for `print` at the end
        of each block.
        """
        print("scaling:", self.scaling)
        print("moment_coeffs:")
        print(self.moment_coeffs)
        print("species_coeffs:")
        print(self.species_coeffs)
        print("radial_coeffs:")
        print(self.radial_coeffs)
        print(**kwargs)
