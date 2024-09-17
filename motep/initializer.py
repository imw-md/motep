"""Initializer."""

import random
from typing import Any

import numpy as np

from motep.pot import generate_random_numbers


def init_parameters(
    data: dict[str, Any],
    optimized: list[str],
    seed: int | None,
) -> tuple[list[float], list[tuple[float, float]]]:
    """Initialize MTP parameters.

    Parameters
    ----------
    data : dict[str, Any]
        Data in the .mtp file.
    optimized : list[str]
        Parameters to be optimized.
    seed : int
        Seed of the pseudo-random-number generator.

    Returns
    -------
    parameters : list[float]
        Initial parameters.
    bounds : list[tuple[float, float]]
        Bounds of the parameters.

    """
    if seed is None:
        seed = random.randrange(2**31 - 1)
    parameters_scaling, bounds_scaling = _init_scaling(
        data,
        optimized,
        seed,
    )
    parameters_moment_coeffs, bounds_moment_coeffs = _init_moment_coeffs(
        data,
        optimized,
        seed,
    )
    parameters_species_coeffs, bounds_species_coeffs = _init_species_coeffs(
        data,
        optimized,
        seed,
    )
    parameters_radial_coeffs, bounds_radial_coeffs = _init_radial_coeffs(
        data,
        optimized,
        seed,
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
    seed: int,
) -> tuple[list[float], list[tuple[float, float]]]:
    key = "scaling"
    v = data.get(key, 1000.0)
    parameters_scaling = [v]
    bounds_scaling = [(-1000.0, 1000.0)] if key in optimized else [(v, v)]
    return parameters_scaling, bounds_scaling


def _init_moment_coeffs(
    data: dict[str, Any],
    optimized: list[str],
    seed: int,
) -> tuple[list[float], list[tuple[float, float]]]:
    asm = data["alpha_scalar_moments"]
    key = "moment_coeffs"
    v = np.array(data[key]) if key in data else np.full(asm, 5.0)
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
    seed: int,
) -> tuple[list[float], list[tuple[float, float]]]:
    species_count = data["species_count"]
    key = "species_coeffs"
    v = np.array(data[key]) if key in data else np.full(species_count, 5.0)
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
    seed: int,
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
        v = np.array(generate_random_numbers(n, lb, ub, seed))
    parameters = v.tolist()
    if key in optimized:
        lb, ub = -0.1, +0.1
        bounds = [(lb, ub)] * n
    else:
        bounds = np.repeat(v[:, None], 2, axis=1).tolist()
    return parameters, bounds
