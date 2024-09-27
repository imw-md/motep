"""Module for the optimizer based on linear least squares (LLS)."""

from typing import Any

import numpy as np
from ase import Atoms

from motep.loss_function import LossFunction, update_mtp
from motep.optimizers.scipy import Callback


class LLSOptimizer:
    """Optimizer based on linear least squares (LLS)."""

    def __init__(self, data: dict[str, Any]) -> None:
        """Initialize the optimizer."""
        self.data = data

    def __call__(
        self,
        fitness: LossFunction,
        parameters: np.ndarray,
        bounds: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Optimize parameters.

        Parameters
        ----------
        fitness : :class:`~motep.loss_function.LossFunction`
            :class:`motep.loss_function.LossFunction` object.
        parameters : np.ndarray
            Initial parameters.
        bounds : np.ndarray
            Lower and upper bounds for the parameters.
            Not used in :class:`~motep.optimizers.lls.LLSOptimizer`.

        Returns
        -------
        parameters : np.ndarray
            Optimized parameters.

        """
        # Calculate basis functions of `fitness.images`
        fitness(parameters)

        # Update self.data based on the initialized parameters
        self.data = update_mtp(self.data, parameters)

        if "species" not in self.data:
            species = {_: _ for _ in range(self.data["species_count"])}
            self.data["species"] = species

        energies = self._calc_interaction_energies(fitness.images, species)

        basis_values = np.array(
            [atoms.calc.engine.basis_values for atoms in fitness.images],
        )

        # TODO: Consider also forces and stresses
        moment_coeffs = np.linalg.lstsq(basis_values, energies, rcond=None)[0]

        # TODO: Redesign optimizers to du such an assignment more semantically
        parameters[1 : len(moment_coeffs) + 1] = moment_coeffs

        # Print loss function value
        Callback(fitness)(parameters)

        return parameters

    def _calc_interaction_energies(
        self,
        images: list[Atoms],
        species: list[int],
    ) -> np.ndarray:
        """Calculate interaction energies of Atoms objects.

        Parameters
        ----------
        images : list[Atoms]
            List of ASE Atoms objects.
        species : dict[int, int]
            Mapping of species to atomic types in the MLIP .mtp file.

        Returns
        -------
        np.ndarray
            Array of interaction energies of the Atoms objects caused by
            interactions among atoms, i.e., without site energies.

        """

        def get_types(atoms: Atoms) -> list[int]:
            return [species[_] for _ in atoms.numbers]

        iterable = (
            np.add.reduce(self.data["species_coeffs"][get_types(atoms)])
            - atoms.get_potential_energy()
            for atoms in images
        )
        return np.fromiter(iterable, dtype=float, count=len(images))
