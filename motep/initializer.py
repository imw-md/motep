"""Initializer."""

from typing import Any

import numpy as np


class Initializer:
    """Class to initialize MTP parameters."""

    def __init__(self, rng: np.random.Generator | int | None) -> None:
        """Initialize Initializer.

        Parameters
        ----------
        rng : np.random.Generator | int | None, default = None
            Pseudo-random-number generator (PRNG) with the NumPy API.
            If ``int`` or ``None``, they are treated as the seed of the NumPy
            default PRNG.

        """
        if isinstance(rng, int | None):
            self.rng = np.random.default_rng(rng)
        else:
            self.rng = rng

    def initialize(
        self,
        data: dict[str, Any],
        optimized: list[str],
    ) -> tuple[list[float], list[tuple[float, float]]]:
        """Initialize MTP parameters.

        Parameters
        ----------
        data : dict[str, Any]
            Data in the .mtp file.
        optimized : list[str]
            Parameters to be optimized.

        Returns
        -------
        parameters : list[float]
            Initial parameters.
        bounds : list[tuple[float, float]]
            Bounds of the parameters.

        """
        parameters_scaling, bounds_scaling = _init_scaling(data, optimized)
        parameters_moment_coeffs, bounds_moment_coeffs = _init_moment_coeffs(
            data,
            optimized,
            self.rng,
        )
        parameters_species_coeffs, bounds_species_coeffs = _init_species_coeffs(
            data,
            optimized,
            self.rng,
        )
        parameters_radial_coeffs, bounds_radial_coeffs = _init_radial_coeffs(
            data,
            optimized,
            self.rng,
        )
        parameters = (
            parameters_scaling
            + parameters_moment_coeffs
            + parameters_species_coeffs
            + parameters_radial_coeffs
        )
        bounds = (
            bounds_scaling
            + bounds_moment_coeffs
            + bounds_species_coeffs
            + bounds_radial_coeffs
        )
        return parameters, bounds


def _init_scaling(
    data: dict[str, Any],
    optimized: list[str],
) -> tuple[list[float], list[tuple[float, float]]]:
    key = "scaling"
    v = data.get(key, 1.0)
    parameters_scaling = [v]
    bounds_scaling = [(0.0, 1000.0)] if key in optimized else [(v, v)]
    return parameters_scaling, bounds_scaling


def _init_moment_coeffs(
    data: dict[str, Any],
    optimized: list[str],
    rng: np.random.Generator,
) -> tuple[list[float], list[tuple[float, float]]]:
    asm = data["alpha_scalar_moments"]
    key = "moment_coeffs"
    if key in data:
        v = np.array(data[key])
    else:
        lb, ub = -5.0, +5.0
        v = rng.uniform(lb, ub, asm)
    parameters = v.tolist()
    if key in optimized:
        lb, ub = -5.0, +5.0
        bounds = [(lb, ub)] * asm
    else:
        bounds = np.repeat(v[:, None], 2, axis=1).tolist()
    return parameters, bounds


def _init_species_coeffs(
    data: dict[str, Any],
    optimized: list[str],
    rng: np.random.Generator,
) -> tuple[list[float], list[tuple[float, float]]]:
    species_count = data["species_count"]
    key = "species_coeffs"
    v = np.array(data[key]) if key in data else np.zeros(species_count)
    parameters = v.tolist()
    if key in optimized:
        lb, ub = -5.0, +5.0
        bounds = [(lb, ub)] * species_count
    else:
        bounds = np.repeat(v[:, None], 2, axis=1).tolist()
    return parameters, bounds


def _init_radial_coeffs(
    data: dict[str, Any],
    optimized: list[str],
    rng: np.random.Generator,
) -> tuple[list[float], list[tuple[float, float]]]:
    species_count = data["species_count"]
    rfc = data["radial_funcs_count"]
    rbs = data["radial_basis_size"]
    n = species_count * species_count * rfc * rbs
    key = "radial_coeffs"
    if key in data:
        v = np.array(data[key]).flatten()
    else:
        lb, ub = -0.1, +0.1
        v = rng.uniform(lb, ub, n)
    parameters = v.tolist()
    if key in optimized:
        lb, ub = -0.1, +0.1
        bounds = [(lb, ub)] * n
    else:
        bounds = np.repeat(v[:, None], 2, axis=1).tolist()
    return parameters, bounds
