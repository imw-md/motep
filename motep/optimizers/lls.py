"""Module for the optimizer based on linear least squares (LLS)."""

from typing import Any

import numpy as np
from ase import Atoms

from motep.optimizers import OptimizerBase
from motep.optimizers.scipy import Callback


class LLSOptimizer(OptimizerBase):
    """Optimizer based on linear least squares (LLS).

    Attributes
    ----------
    minimized : list[str]
        Properties whose errors are minimized by optimizing `moment_coeffs`.
        The elements must be some of `energy`, `forces`, and `stress`.

    """

    def __init__(
        self,
        *args: list[Any],
        minimized: list[str] | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__(*args, **kwargs)
        if minimized is None:
            minimized = ["energy", "forces"]
        self.minimized = minimized

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

        # TODO: Consider also `species_coeffs`
        moment_coeffs = np.linalg.lstsq(matrix, vector, rcond=None)[0]

        # TODO: Redesign optimizers to du such an assignment more semantically
        parameters[1 : len(moment_coeffs) + 1] = moment_coeffs

        # Print the value of the loss function.
        callback(parameters)

        return parameters

    def _calc_matrix(self) -> np.ndarray:
        """Calculate the matrix for linear least squares (LLS)."""
        loss = self.loss_function
        dict_mtp = loss.mtp_data.dict_mtp
        images = loss.images
        setting = loss.setting
        basis_values = np.array([atoms.calc.engine.basis_values for atoms in images])
        basis_dbdris = np.vstack([atoms.calc.engine.basis_dbdris.T for atoms in images])
        basis_dbdeps = np.vstack([atoms.calc.engine.basis_dbdeps.T for atoms in images])
        basis_dbdris = basis_dbdris.reshape((-1, dict_mtp["alpha_scalar_moments"]))
        basis_dbdeps = basis_dbdeps.reshape((-1, dict_mtp["alpha_scalar_moments"]))
        tmp = []
        if "energy" in self.minimized:
            tmp.append(np.sqrt(setting["energy-weight"]) * basis_values)
        if "forces" in self.minimized:
            tmp.append(np.sqrt(setting["force-weight"]) * basis_dbdris)
        if "stress" in self.minimized:
            tmp.append(np.sqrt(setting["stress-weight"]) * basis_dbdeps)
        return np.vstack(tmp)

    def _calc_vector(self) -> np.ndarray:
        """Calculate the vector for linear least squares (LLS)."""
        loss = self.loss_function
        setting = loss.setting
        n = len(loss.images)
        energies = self._calc_interaction_energies()
        forces = np.hstack([loss.target_forces[i].flatten() for i in range(n)])
        stresses = np.hstack([loss.target_stresses[i].flatten() for i in range(n)])
        tmp = []
        if "energy" in self.minimized:
            tmp.append(np.sqrt(setting["energy-weight"]) * energies)
        if "forces" in self.minimized:
            tmp.append(np.sqrt(setting["force-weight"]) * forces * -1.0)
        if "stress" in self.minimized:
            tmp.append(np.sqrt(setting["stress-weight"]) * stresses)
        return np.hstack(tmp)

    def _calc_interaction_energies(self) -> np.ndarray:
        """Calculate interaction energies of Atoms objects.

        Returns
        -------
        np.ndarray
            Array of interaction energies of the Atoms objects caused by
            interactions among atoms, i.e., without site energies.

        """
        loss = self.loss_function
        dict_mtp = loss.mtp_data.dict_mtp
        images = loss.images
        species = dict_mtp["species"]

        def get_types(atoms: Atoms) -> list[int]:
            return [species[_] for _ in atoms.numbers]

        iterable = (
            loss.target_energies[i]
            - np.add.reduce(dict_mtp["species_coeffs"][get_types(atoms)])
            for i, atoms in enumerate(images)
        )
        return np.fromiter(iterable, dtype=float, count=len(images))
