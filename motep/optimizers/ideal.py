"""Module for `NoInteractionOptimizer`."""

from typing import Any

import numpy as np

from motep.optimizers.base import OptimizerBase
from motep.optimizers.scipy import Callback


class NoInteractionOptimizer(OptimizerBase):
    """Optimizer assuming no atomic interaction."""

    def optimize(
        self,
        parameters: np.ndarray,
        bounds: np.ndarray,
        **kwargs: dict[str, Any],
    ) -> np.ndarray:
        """Optimize `species_coeffs`.

        The values are determined using the least-square method.
        Note that, if there are no composition varieties in the training set,
        the values are physically less meaningful.

        Parameters
        ----------
        parameters : np.ndarray
            Initial parameters.
        bounds : np.ndarray
            Lower and upper bounds for the parameters.
        **kwargs : dict[str, Any]
            Other keyward arguments passed to the actual optimization function.

        Returns
        -------
        parameters : np.ndarray
            Optimized parameters.

        """
        # Calculate basis functions of `fitness.images`
        self.loss_function(parameters)

        callback = Callback(self.loss_function)

        # Print the value of the loss function.
        callback(parameters)

        # Update self.data based on the initialized parameters
        self.loss_function.mtp_data.update(parameters)

        matrix = self._calc_matrix()
        vector = self._calc_vector()

        species_coeffs = np.linalg.lstsq(matrix, vector, rcond=None)[0]

        self.loss_function.mtp_data.dict_mtp["scaling"] = 1.0
        self.loss_function.mtp_data.dict_mtp["moment_coeffs"][...] = 0.0
        self.loss_function.mtp_data.dict_mtp["radial_coeffs"][...] = 0.0
        self.loss_function.mtp_data.dict_mtp["species_coeffs"] = species_coeffs

        parameters = self.loss_function.mtp_data.parameters

        # Print the value of the loss function.
        callback(parameters)

        return parameters

    def _calc_matrix(self) -> np.ndarray:
        """Calculate the matrix for linear least squares (LLS)."""
        loss = self.loss_function
        species = loss.mtp_data.dict_mtp["species"]
        images = loss.images
        counts = np.full((len(images), len(species)), np.nan)
        for i, atoms in enumerate(images):
            for j, s in enumerate(species):
                counts[i, j] = list(atoms.numbers).count(s)
        return counts

    def _calc_vector(self) -> np.ndarray:
        """Calculate the vector for linear least squares (LLS)."""
        n = len(self.loss_function.images)
        iterable = (self.loss_function.target_energies[i] for i in range(n))
        return np.fromiter(iterable, dtype=float, count=n)
