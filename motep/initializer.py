"""Initializer."""

from typing import Any

import numpy as np
import numpy.typing as npt
from ase import Atoms


class Initializer:
    """Class to initialize MTP parameters."""

    def __init__(
        self,
        images: list[Atoms],
        species: list[str],
        rng: np.random.Generator | int | None,
    ) -> None:
        """Initialize Initializer.

        Parameters
        ----------
        images : list[Atoms]
            List of ASE Atoms objects for the training dataset.
            They are used to determine the initial guess of `species_coeffs`.
        species : list[str]
            List of species in the order of `type` in the MLIP cfg file.
        rng : np.random.Generator | int | None, default = None
            Pseudo-random-number generator (PRNG) with the NumPy API.
            If ``int`` or ``None``, they are treated as the seed of the NumPy
            default PRNG.

        """
        self.species_coeffs_lstsq = calc_species_coeffs_lstsq(images, species)
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
        print("scaling:", *parameters_scaling)
        parameters_moment_coeffs, bounds_moment_coeffs = _init_moment_coeffs(
            data,
            optimized,
            self.rng,
        )
        parameters_species_coeffs, bounds_species_coeffs = _init_species_coeffs(
            data,
            self.species_coeffs_lstsq,
            optimized,
        )
        print("species_coeffs:", parameters_species_coeffs)
        parameters_radial_coeffs, bounds_radial_coeffs = _init_radial_coeffs(
            data,
            optimized,
            self.rng,
        )
        parameters = np.hstack(
            (
                parameters_scaling,
                parameters_moment_coeffs,
                parameters_species_coeffs,
                parameters_radial_coeffs.reshape(-1),
            ),
        )
        bounds = np.vstack(
            (
                bounds_scaling,
                bounds_moment_coeffs,
                bounds_species_coeffs,
                bounds_radial_coeffs.reshape(-1, 2),
            ),
        )
        return parameters, bounds


def calc_species_coeffs_lstsq(
    images: list[Atoms],
    species: list[str],
) -> npt.NDArray[np.float64]:
    """Calculate `species_coeffs` assuming no interatomic forces.

    The values are determined using the least-square method.
    Note that, if there are no composition varieties in the training set,
    the values are physically less meaningful.
    """
    counts = np.full((len(images), len(species)), np.nan)
    energies = np.full(len(images), np.nan)
    for i, atoms in enumerate(images):
        for j, s in enumerate(species):
            counts[i, j] = atoms.symbols.count(s)
        energies[i] = atoms.get_potential_energy(force_consistent=True)
    ns = counts.sum(axis=1)
    counts /= ns[:, None]
    energies /= ns
    species_coeffs_lstsq = np.linalg.lstsq(counts, energies, rcond=None)[0]
    rmse = np.sqrt(np.add.reduce((counts @ species_coeffs_lstsq - energies) ** 2))
    print("RMSE Energy per atom (eV/atom):", rmse)
    return species_coeffs_lstsq


def _init_scaling(
    data: dict[str, Any],
    optimized: list[str],
) -> tuple[list[float], list[tuple[float, float]]]:
    key = "scaling"
    v = data.get(key, 1.0)
    parameters_scaling = np.array([v])
    bounds_scaling = np.array([(0.0, 1e6)] if key in optimized else [(v, v)])
    return parameters_scaling, bounds_scaling


def _init_moment_coeffs(
    data: dict[str, Any],
    optimized: list[str],
    rng: np.random.Generator,
) -> tuple[list[float], list[tuple[float, float]]]:
    asm = data["alpha_scalar_moments"]
    key = "moment_coeffs"
    if key in data:
        parameters = np.asarray(data[key])
    else:
        lb, ub = -5.0, +5.0
        parameters = rng.uniform(lb, ub, asm)
    if key in optimized:
        lb, ub = -5.0, +5.0
        bounds = [(lb, ub)] * asm
    else:
        bounds = np.repeat(parameters[:, None], 2, axis=1)
    return parameters, bounds


def _init_species_coeffs(
    data: dict[str, Any],
    species_coeffs_lstsq: npt.NDArray[np.float64],
    optimized: list[str],
) -> tuple[list[float], list[tuple[float, float]]]:
    key = "species_coeffs"
    parameters = np.asarray(data[key]) if key in data else species_coeffs_lstsq
    bounds = np.repeat(parameters[:, None], 2, axis=1)
    if key in optimized:
        w = 5.0
        bounds[:, 0] -= w
        bounds[:, 1] += w
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
        parameters = np.asarray(data[key])
    else:
        lb, ub = -0.1, +0.1
        parameters = rng.uniform(lb, ub, n)
    if key in optimized:
        lb, ub = -0.1, +0.1
        bounds = np.array([(lb, ub)] * n)
    else:
        bounds = np.repeat(parameters[:, None], 2, axis=1)
    return parameters, bounds
