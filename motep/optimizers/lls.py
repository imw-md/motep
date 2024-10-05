"""Module for the optimizer based on linear least squares (LLS)."""

import numpy as np
from ase import Atoms

from motep.optimizers import OptimizerBase
from motep.optimizers.scipy import Callback


class LLSOptimizer(OptimizerBase):
    """Optimizer based on linear least squares (LLS)."""

    def optimize(
        self,
        parameters: np.ndarray,
        bounds: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Optimize parameters.

        Parameters
        ----------
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
        self.loss_function(parameters)

        callback = Callback(self.loss_function)

        # Print the value of the loss function.
        callback(parameters)

        # Update self.data based on the initialized parameters
        self.loss_function.mtp_data.update(parameters)

        matrix = self._calc_matrix()
        vector = self._calc_vector()

        # TODO: Consider also forces and stresses
        moment_coeffs = np.linalg.lstsq(matrix, vector, rcond=None)[0]

        # TODO: Redesign optimizers to du such an assignment more semantically
        parameters[1 : len(moment_coeffs) + 1] = moment_coeffs

        # Print the value of the loss function.
        callback(parameters)

        return parameters

    def _calc_matrix(self) -> np.ndarray:
        """Calculate the matrix for linear least squares (LLS)."""
        loss_function = self.loss_function
        dict_mtp = loss_function.mtp_data.dict_mtp
        images = loss_function.images
        setting = loss_function.setting
        basis_values = np.array([atoms.calc.engine.basis_values for atoms in images])
        basis_derivs = np.vstack([atoms.calc.engine.basis_derivs.T for atoms in images])
        basis_derivs = basis_derivs.reshape((-1, dict_mtp["alpha_scalar_moments"]))
        tmp = (
            np.sqrt(setting["energy-weight"]) * basis_values,
            np.sqrt(setting["force-weight"]) * basis_derivs,
        )
        return np.vstack(tmp)

    def _calc_vector(self) -> np.ndarray:
        """Calculate the vector for linear least squares (LLS)."""
        loss_function = self.loss_function
        images = loss_function.images
        setting = loss_function.setting
        energies = self._calc_interaction_energies()
        forces = np.hstack(
            [
                loss_function.target_forces[i].reshape(-1)
                for i, atoms in enumerate(images)
            ],
        )
        tmp = (
            np.sqrt(setting["energy-weight"]) * energies,
            np.sqrt(setting["force-weight"]) * forces * -1.0,
        )
        return np.hstack(tmp)

    def _calc_interaction_energies(self) -> np.ndarray:
        """Calculate interaction energies of Atoms objects.

        Returns
        -------
        np.ndarray
            Array of interaction energies of the Atoms objects caused by
            interactions among atoms, i.e., without site energies.

        """
        loss_function = self.loss_function
        dict_mtp = loss_function.mtp_data.dict_mtp
        images = loss_function.images
        species = dict_mtp["species"]

        def get_types(atoms: Atoms) -> list[int]:
            return [species[_] for _ in atoms.numbers]

        iterable = (
            loss_function.target_energies[i]
            - np.add.reduce(dict_mtp["species_coeffs"][get_types(atoms)])
            for i, atoms in enumerate(images)
        )
        return np.fromiter(iterable, dtype=float, count=len(images))
