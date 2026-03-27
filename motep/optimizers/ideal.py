"""Module for `NoInteractionOptimizer`."""

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize._optimize import OptimizeResult

from motep.optimizers.base import ParallelOptimizerBase
from motep.optimizers.scipy import Callback


class NoInteractionOptimizer(ParallelOptimizerBase):
    """Optimizer assuming no atomic interaction."""

    def _optimize(self, **kwargs: dict[str, Any]) -> npt.NDArray[np.float64]:
        """Optimize `species_coeffs`.

        The values are determined using the least-square method.
        Note that, if there are no composition varieties in the training set,
        the values are physically less meaningful.

        Returns
        -------
        npt.NDArray[np.float64]

        """
        parameters = self.loss.mtp_data.parameters
        callback = Callback(self.loss)

        # Calculate basis functions of `loss.images`
        loss_value = self.rank0_loss(parameters)
        self.rank0_gather_data()

        # Print the value of the loss function.
        callback(OptimizeResult(x=parameters, fun=loss_value))

        # Prepare and solve the LLS problem
        matrix = self._calc_matrix()
        vector = self._calc_vector()

        species_coeffs = np.linalg.lstsq(matrix, vector, rcond=None)[0]

        # Update `mtp_data`
        self.loss.mtp_data.scaling = 1.0
        self.loss.mtp_data.moment_coeffs[...] = 0.0
        self.loss.mtp_data.radial_coeffs[...] = 0.0
        self.loss.mtp_data.species_coeffs = species_coeffs
        # Update `parameters` by calling the property
        parameters = self.loss.mtp_data.parameters

        # Evaluate loss with the new parameters
        loss_value = self.rank0_loss(parameters)

        # Print the value of the loss function.
        callback(OptimizeResult(x=parameters, fun=loss_value))

        return parameters

    @property
    def optimized_default(self) -> list[str]:
        return ["species_coeffs"]

    @property
    def optimized_allowed(self) -> list[str]:
        return ["species_coeffs"]

    def _calc_matrix(self) -> np.ndarray:
        """Calculate the matrix for linear least squares (LLS).

        Returns
        -------
        np.ndarray

        """
        loss = self.loss
        species = loss.mtp_data.species
        images = loss.images
        counts = np.full((len(images), len(species)), np.nan)
        for i, atoms in enumerate(images):
            for j, s in enumerate(species):
                counts[i, j] = list(atoms.numbers).count(s)
        return counts

    def _calc_vector(self) -> np.ndarray:
        """Calculate the vector for linear least squares (LLS).

        Returns
        -------
        np.ndarray

        """
        images = self.loss.images
        return np.fromiter(
            (atoms.calc.targets["energy"] for atoms in images),
            dtype=float,
            count=len(images),
        )
