"""Module for the optimizer based on linear least squares (LLS)."""

import numpy as np
from ase import Atoms

from motep.initializer import MTPData
from motep.loss_function import LossFunction
from motep.optimizers.scipy import Callback


class LLSOptimizer:
    """Optimizer based on linear least squares (LLS)."""

    def __init__(self, data: MTPData) -> None:
        """Initialize the optimizer."""
        self.data = data
        if "species" not in self.data.data:
            species = {_: _ for _ in range(self.data.data["species_count"])}
            self.data.data["species"] = species

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
        self.data.update(parameters)

        energies = self._calc_interaction_energies(fitness)

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

    def _calc_interaction_energies(self, fitness: LossFunction) -> np.ndarray:
        """Calculate interaction energies of Atoms objects.

        Parameters
        ----------
        fitness : :class:`~motep.loss_function.LossFunction`
            :class:`motep.loss_function.LossFunction` object.

        Returns
        -------
        np.ndarray
            Array of interaction energies of the Atoms objects caused by
            interactions among atoms, i.e., without site energies.

        """
        dict_mtp = self.data.data
        species = dict_mtp["species"]
        images = fitness.images

        def get_types(atoms: Atoms) -> list[int]:
            return [species[_] for _ in atoms.numbers]

        iterable = (
            fitness.target_energies[i]
            - np.add.reduce(dict_mtp["species_coeffs"][get_types(atoms)])
            for i, atoms in enumerate(images)
        )
        return np.fromiter(iterable, dtype=float, count=len(images))
