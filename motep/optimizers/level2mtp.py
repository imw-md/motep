"""Optimizer for Level 2 MTP."""

from typing import Any

import numpy as np
from ase import Atoms

from motep.loss_function import LossFunctionBase
from motep.optimizers.lls import LLSOptimizerBase
from motep.optimizers.scipy import Callback


class Level2MTPOptimizer(LLSOptimizerBase):
    """Optimizer for Level 2 MTP.

    Attributes
    ----------
    optimized : list[str]
        Coefficients to be optimized.
        The elements must be some of `species_coeffs` and `radial_coeffs`.

    """

    def __init__(
        self,
        loss_function: LossFunctionBase,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the optimizer."""
        super().__init__(loss_function, **kwargs)
        if self.optimized is None:
            self.optimized = ["species_coeffs", "radial_coeffs"]

    def optimize(
        self,
        parameters: np.ndarray,
        bounds: np.ndarray,
        **kwargs,
    ) -> None:
        """Optimize parameters.

        Parameters
        ----------
        parameters : np.ndarray
            Initial parameters.
        bounds : np.ndarray
            Lower and upper bounds for the parameters.
            Not used in this class.

        Returns
        -------
        parameters : np.ndarray
            Optimized parameters.

        """
        # Calculate basis functions of `loss_function.images`
        self.loss_function(parameters)

        callback = Callback(self.loss_function)

        # Print the value of the loss function.
        callback(parameters)

        # Update self.data based on the initialized parameters
        self.loss_function.mtp_data.update(parameters)

        matrix = self._calc_matrix()
        vector = self._calc_vector()

        # TODO: Consider also forces and stresses
        coeffs, *_ = np.linalg.lstsq(matrix, vector, rcond=None)

        # Update `dict_mtp` and `parameters`.
        parameters = self._update_parameters(coeffs)

        # Print the value of the loss function.
        callback(parameters)

        return parameters

    def _update_parameters(self, coeffs: np.ndarray) -> np.ndarray:
        mtp_data = self.loss_function.mtp_data
        species_count = mtp_data.dict_mtp["species_count"]
        rbs = mtp_data.dict_mtp["radial_basis_size"]
        size = species_count * species_count * rbs
        shape = species_count, species_count, rbs

        mtp_data.dict_mtp["scaling"] = 1.0
        mtp_data.dict_mtp["moment_coeffs"][...] = 0.0
        mtp_data.dict_mtp["moment_coeffs"][000] = 1.0
        mtp_data.dict_mtp["radial_coeffs"][000] = 0.0
        mtp_data.dict_mtp["radial_coeffs"][:, :, 0, :] = coeffs[:size].reshape(shape)
        if "species_coeffs" in self.optimized:
            mtp_data.dict_mtp["species_coeffs"] = coeffs[size:]

        return mtp_data.parameters

    def _calc_matrix(self) -> np.ndarray:
        """Calculate the matrix for linear least squares (LLS)."""
        tmp = []
        tmp.append(self._calc_matrix_radial_coeffs())
        if "species_coeffs" in self.optimized:
            tmp.append(self._calc_matrix_species_coeffs())
        return np.hstack(tmp)

    def _calc_matrix_radial_coeffs(self) -> np.ndarray:
        loss = self.loss_function
        dict_mtp = loss.mtp_data.dict_mtp
        images = loss.images
        setting = loss.setting
        species_count = dict_mtp["species_count"]
        radial_basis_size = dict_mtp["radial_basis_size"]
        size = species_count * species_count * radial_basis_size

        def get_radial_basis_values(atoms: Atoms) -> np.ndarray:
            return atoms.calc.engine.radial_basis_values[:, :, :]

        def get_radial_basis_derivs(atoms: Atoms) -> np.ndarray:
            return atoms.calc.engine.radial_basis_derivs[:, :, :, :, :].T

        values = np.stack([get_radial_basis_values(atoms) for atoms in images])
        derivs = np.stack([get_radial_basis_derivs(atoms) for atoms in images])
        values = values.reshape(-1, size)
        derivs = derivs.reshape(-1, size)
        tmp = []
        if "energy" in self.minimized:
            tmp.append(np.sqrt(setting["energy-weight"]) * values)
        if "forces" in self.minimized:
            tmp.append(np.sqrt(setting["force-weight"]) * derivs)
        # if "stress" in self.minimized:
        #     tmp.append(np.sqrt(setting["stress-weight"]) * dbdeps)
        return np.vstack(tmp)
