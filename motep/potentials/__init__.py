"""Initializer."""

from typing import Any

import numpy as np
import numpy.typing as npt
from ase import Atoms


class MTPData:
    """Class to handle MTP parameters."""

    def __init__(
        self,
        dict_mtp: dict[str, Any],
        images: list[Atoms],
        species: list[str],
        rng: np.random.Generator | int | None,
    ) -> None:
        """Initialize Initializer.

        Parameters
        ----------
        dict_mtp : dict[str, Any]
            Data in the .mtp file.
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
        self.dict_mtp = dict_mtp
        self.species_coeffs_lstsq = calc_species_coeffs_lstsq(images, species)
        if isinstance(rng, int | None):
            self.rng = np.random.default_rng(rng)
        else:
            self.rng = rng

    def initialize(self, optimized: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Initialize MTP parameters.

        Parameters
        ----------
        optimized : list[str]
            Parameters to be optimized.

        Returns
        -------
        parameters : list[float]
            Initial parameters.
        bounds : list[tuple[float, float]]
            Bounds of the parameters.

        """
        dict_mtp = self.dict_mtp
        scaling, bound_scaling = _init_scaling(dict_mtp, optimized)
        moment_coeffs, bounds_moment_coeffs = _init_moment_coeffs(
            dict_mtp,
            optimized,
            self.rng,
        )
        species_coeffs, bounds_species_coeffs = _init_species_coeffs(
            dict_mtp,
            self.species_coeffs_lstsq,
            optimized,
        )
        radial_coeffs, bounds_radial_coeffs = _init_radial_coeffs(
            dict_mtp,
            optimized,
            self.rng,
        )
        dict_mtp["scaling"] = scaling
        dict_mtp["moment_coeffs"] = moment_coeffs
        dict_mtp["species_coeffs"] = species_coeffs
        dict_mtp["radial_coeffs"] = radial_coeffs
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
        return np.hstack(
            (
                np.atleast_1d(self.dict_mtp["scaling"]),
                self.dict_mtp["moment_coeffs"],
                self.dict_mtp["species_coeffs"],
                self.dict_mtp["radial_coeffs"].reshape(-1),
            ),
        )

    def update(self, parameters: list[float]) -> None:
        """Update data in the .mtp file.

        Parameters
        ----------
        parameters : list[float]
            MTP parameters.

        """
        dict_mtp = self.dict_mtp
        species_count = dict_mtp["species_count"]
        rfc = dict_mtp["radial_funcs_count"]
        rbs = dict_mtp["radial_basis_size"]
        asm = dict_mtp["alpha_scalar_moments"]

        dict_mtp["scaling"] = parameters[0]
        dict_mtp["moment_coeffs"] = parameters[1 : asm + 1]
        dict_mtp["species_coeffs"] = parameters[asm + 1 : asm + 1 + species_count]
        total_radial = parameters[asm + 1 + species_count :]
        shape = species_count, species_count, rfc, rbs
        dict_mtp["radial_coeffs"] = np.array(total_radial).reshape(shape)

    def print(self) -> None:
        """Print parameters."""
        print("scaling:", self.dict_mtp["scaling"])
        print("moment_coeffs:")
        print(self.dict_mtp["moment_coeffs"])
        print("species_coeffs:")
        print(self.dict_mtp["species_coeffs"])
        print("radial_coeffs:")
        print(self.dict_mtp["radial_coeffs"])
        print()


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
    print(f"{__name__}.{calc_species_coeffs_lstsq.__name__}")
    print("species_coeffs_lstsq:")
    print(species_coeffs_lstsq)
    print("RMSE Energy per atom (eV/atom):", rmse)
    print("The values are estimated assuming no interatomic forces.")
    print()
    return species_coeffs_lstsq


def _init_scaling(
    data: dict[str, Any],
    optimized: list[str],
) -> tuple[float, tuple[float, float]]:
    key = "scaling"
    v = data.get(key, 1.0)
    parameters_scaling = v
    bounds_scaling = (0.0, np.inf) if key in optimized else (v, v)
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
        bounds = [(-np.inf, +np.inf)] * asm
    else:
        bounds = np.repeat(parameters[:, None], 2, axis=1)
    return parameters, bounds


def _init_species_coeffs(
    data: dict[str, Any],
    species_coeffs_lstsq: npt.NDArray[np.float64],
    optimized: list[str],
) -> tuple[list[float], list[tuple[float, float]]]:
    species_count = data["species_count"]
    key = "species_coeffs"
    parameters = np.asarray(data[key]) if key in data else species_coeffs_lstsq
    if key in optimized:
        bounds = np.array([(-np.inf, +np.inf)] * species_count)
    else:
        bounds = np.repeat(parameters[:, None], 2, axis=1)
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
        bounds = np.array([(-np.inf, +np.inf)] * n)
    else:
        bounds = np.repeat(parameters[:, None], 2, axis=1)
    return parameters, bounds
