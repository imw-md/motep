"""Initializer."""

from dataclasses import dataclass, field

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

    def initialize(
        self,
        optimized: list[str],
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Initialize MTP parameters.

        Parameters
        ----------
        optimized : list[str]
            Parameters to be optimized.
        rng : np.random.Generator
            Pseudo-random-number generator (PRNG) with the NumPy API.

        Returns
        -------
        parameters : list[float]
            Initial parameters.
        bounds : list[tuple[float, float]]
            Bounds of the parameters.

        """
        scaling, bound_scaling = _init_scaling(self, optimized)
        species_coeffs, bounds_species_coeffs = _init_species_coeffs(
            self,
            optimized,
            rng,
        )
        moment_coeffs, bounds_moment_coeffs = _init_moment_coeffs(
            self,
            optimized,
            rng,
        )
        radial_coeffs, bounds_radial_coeffs = _init_radial_coeffs(
            self,
            optimized,
            rng,
        )
        self.scaling = scaling
        self.moment_coeffs = moment_coeffs
        self.species_coeffs = species_coeffs
        self.radial_coeffs = radial_coeffs
        bounds = np.vstack(
            (
                np.atleast_1d(bound_scaling),
                bounds_moment_coeffs,
                bounds_species_coeffs,
                bounds_radial_coeffs.reshape(-1, 2),
            ),
        )
        return self.parameters, bounds

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

    def print(self) -> None:
        """Print parameters."""
        print("scaling:", self.scaling)
        print("moment_coeffs:")
        print(self.moment_coeffs)
        print("species_coeffs:")
        print(self.species_coeffs)
        print("radial_coeffs:")
        print(self.radial_coeffs)
        print()


def _init_scaling(
    data: MTPData,
    optimized: list[str],
) -> tuple[float, tuple[float, float]]:
    v = data.scaling
    parameters_scaling = v
    bounds_scaling = (0.0, np.inf) if "scaling" in optimized else (v, v)
    return parameters_scaling, bounds_scaling


def _init_moment_coeffs(
    data: MTPData,
    optimized: list[str],
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    asm = data.alpha_scalar_moments
    if data.moment_coeffs is not None:
        parameters = np.asarray(data.moment_coeffs)
    else:
        lb, ub = -5.0, +5.0
        parameters = rng.uniform(lb, ub, asm)
    if "moment_coeffs" in optimized:
        bounds = np.array([(-np.inf, +np.inf)] * asm)
    else:
        bounds = np.repeat(parameters[:, None], 2, axis=1)
    return parameters, bounds


def _init_species_coeffs(
    data: MTPData,
    optimized: list[str],
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    species_count = data.species_count
    if data.species_coeffs is not None:
        parameters = np.asarray(data.species_coeffs)
    else:
        lb, ub = -5.0, +5.0
        parameters = rng.uniform(lb, ub, species_count)
    if "species_coeffs" in optimized:
        bounds = np.array([(-np.inf, +np.inf)] * species_count)
    else:
        bounds = np.repeat(parameters[:, None], 2, axis=1)
    return parameters, bounds


def _init_radial_coeffs(
    data: MTPData,
    optimized: list[str],
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    species_count = data.species_count
    rfc = data.radial_funcs_count
    rbs = data.radial_basis_size
    n = species_count * species_count * rfc * rbs
    if data.radial_coeffs is not None:
        parameters = np.asarray(data.radial_coeffs)
    else:
        lb, ub = -0.1, +0.1
        parameters = rng.uniform(lb, ub, n)
    if "radial_coeffs" in optimized:
        bounds = np.array([(-np.inf, +np.inf)] * n)
    else:
        bounds = np.repeat(parameters[:, None], 2, axis=1)
    return parameters, bounds
