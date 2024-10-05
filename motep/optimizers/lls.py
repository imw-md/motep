"""Module for the optimizer based on linear least squares (LLS)."""

import numpy as np
from ase import Atoms

from motep.loss_function import LossFunction
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

        matrix = self._calc_matrix(self.loss_function)
        vector = self._calc_vector(self.loss_function)

        # TODO: Consider also forces and stresses
        moment_coeffs = np.linalg.lstsq(matrix, vector, rcond=None)[0]

        # TODO: Redesign optimizers to du such an assignment more semantically
        parameters[1 : len(moment_coeffs) + 1] = moment_coeffs

        # Print the value of the loss function.
        callback(parameters)

        return parameters

    def _calc_matrix(self, fitness: LossFunction) -> np.ndarray:
        """Calculate the matrix for linear least squares (LLS)."""
        dict_mtp = self.loss_function.mtp_data.dict_mtp
        images = fitness.images
        basis_values = np.array([atoms.calc.engine.basis_values for atoms in images])
        basis_derivs = np.vstack([atoms.calc.engine.basis_derivs.T for atoms in images])
        basis_derivs = basis_derivs.reshape((-1, dict_mtp["alpha_scalar_moments"]))
        tmp = (
            np.sqrt(fitness.setting["energy-weight"]) * basis_values,
            np.sqrt(fitness.setting["force-weight"]) * basis_derivs,
        )
        return np.vstack(tmp)

    def _calc_vector(self, fitness: LossFunction) -> np.ndarray:
        """Calculate the vector for linear least squares (LLS)."""
        images = fitness.images
        energies = self._calc_interaction_energies(fitness)
        forces = np.hstack(
            [fitness.target_forces[i].reshape(-1) for i, atoms in enumerate(images)]
        )
        tmp = (
            np.sqrt(fitness.setting["energy-weight"]) * energies,
            np.sqrt(fitness.setting["force-weight"]) * forces * -1.0,
        )
        return np.hstack(tmp)

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
        dict_mtp = self.loss_function.mtp_data.dict_mtp
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
