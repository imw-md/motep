"""Module for the optimizer based on linear least squares (LLS)."""

from abc import abstractmethod
from typing import Any

import numpy as np
from ase.stress import voigt_6_to_full_3x3_stress

from motep.loss import LossFunctionBase
from motep.optimizers.base import OptimizerBase
from motep.optimizers.scipy import Callback
from motep.potentials.mtp.numpy import get_types


class LLSOptimizerBase(OptimizerBase):
    """Abstract base class for `LLSOptimizer` and `Level2MTPOptimizer`.

    Attributes
    ----------
    minimized : list[str]
        Properties whose errors are minimized by optimizing `radial_coeffs`.
        The elements must be some of `energy`, `forces`, and `stress`.

    """

    def __init__(
        self,
        loss: LossFunctionBase,
        *,
        minimized: list[str] | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the optimizer."""
        super().__init__(loss=loss, **kwargs)
        if minimized is None:
            minimized = ["energy", "forces"]
        self.minimized = minimized

    @abstractmethod
    def _calc_matrix(self) -> np.ndarray:
        """Calculate the matrix for linear least squares (LLS)."""

    def _calc_matrix_species_coeffs(self) -> np.ndarray:
        loss = self.loss
        images = loss.images
        species = loss.mtp_data["species"]
        setting = loss.setting
        tmp = []
        if "energy" in self.minimized:
            v = self._calc_matrix_energies_species_coeffs()
            tmp.append(np.sqrt(setting["energy-weight"]) * v)
        if "forces" in self.minimized:
            shape = (
                sum(atoms.calc.results["forces"].size for atoms in images),
                len(species),
            )
            tmp.append(np.zeros(shape))
        if "stress" in self.minimized:
            shape = (9 * len(images), len(species))
            tmp.append(np.zeros(shape))
        return np.vstack(tmp)

    def _calc_matrix_energies_species_coeffs(self) -> np.ndarray:
        loss = self.loss
        species = loss.mtp_data["species"]
        images = loss.images
        counts = np.full((len(images), len(species)), np.nan)
        for i, atoms in enumerate(images):
            for j, s in enumerate(species):
                counts[i, j] = list(atoms.numbers).count(s)
        return counts

    def _calc_vector(self) -> np.ndarray:
        """Calculate the vector for linear least squares (LLS)."""
        setting = self.loss.setting
        images = self.loss.images
        energies = self._calc_energies()
        tmp = []
        if "energy" in self.minimized:
            tmp.append(np.sqrt(setting["energy-weight"]) * energies)
        if "forces" in self.minimized:
            forces = np.hstack([atoms.calc.targets["forces"].flat for atoms in images])
            tmp.append(np.sqrt(setting["force-weight"]) * forces * -1.0)
        if "stress" in self.minimized:
            tmp.append(np.sqrt(setting["stress-weight"]) * self._calc_vector_stress())
        return np.hstack(tmp)

    def _calc_energies(self) -> np.ndarray:
        """Calculate energies of Atoms objects.

        Returns
        -------
        energies : np.ndarray
            Array of interaction energies of the Atoms objects.
            If the key `species_coeffs` is not in `optimized`, this is the
            energies due to interactions among atoms without site energies.
            Otherwise, this is the raw energies including site energies.

        """
        loss = self.loss
        mtp_data = loss.mtp_data
        images = loss.images
        species: list[int] = mtp_data["species"]

        energies = self._calc_target_energies()
        if "species_coeffs" not in self.optimized:
            iterable = (
                np.add.reduce(mtp_data["species_coeffs"][get_types(atoms, species)])
                for atoms in images
            )
            energies -= np.fromiter(iterable, dtype=float, count=len(images))
        return energies

    def _calc_target_energies(self) -> np.ndarray:
        """Calculate the target energies."""
        images = self.loss.images
        return np.fromiter(
            (atoms.calc.targets["energy"] for atoms in images),
            dtype=float,
            count=len(images),
        )

    def _calc_vector_stress(self) -> np.ndarray:
        key = "stress"
        images = self.loss.images
        idcs_str = self.loss.idcs_str
        f = voigt_6_to_full_3x3_stress
        stresses = np.array([f(images[i].calc.targets[key]) for i in idcs_str])
        if self.loss.setting.get("stress-times-volume"):
            stresses = (stresses.T * self.loss.volumes[idcs_str]).T
        return stresses.flat


class LLSOptimizer(LLSOptimizerBase):
    """Optimizer based on linear least squares (LLS).

    Attributes
    ----------
    optimized : list[str]
        Coefficients to be optimized.
        The elements must be some of `species_coeffs` and `moment_coeffs`.

    """

    @property
    def optimized_default(self) -> list[str]:
        return ["species_coeffs", "moment_coeffs"]

    @property
    def optimized_allowed(self) -> list[str]:
        return ["species_coeffs", "moment_coeffs"]

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
        # Calculate basis functions of `loss.images`
        self.loss(parameters)

        callback = Callback(self.loss)

        # Print the value of the loss function.
        callback(parameters)

        # Update self.data based on the initialized parameters
        self.loss.mtp_data.parameters = parameters

        matrix = self._calc_matrix()
        vector = self._calc_vector()

        coeffs = np.linalg.lstsq(matrix, vector, rcond=None)[0]

        # Update `dict_mtp` and `parameters`.
        asm = self.loss.mtp_data["alpha_scalar_moments"]
        self.loss.mtp_data["moment_coeffs"] = coeffs[:asm]
        if "species_coeffs" in self.optimized:
            self.loss.mtp_data["species_coeffs"] = coeffs[asm:]
        parameters = self.loss.mtp_data.parameters

        # Print the value of the loss function.
        callback(parameters)

        return parameters

    def _calc_matrix(self) -> np.ndarray:
        """Calculate the matrix for linear least squares (LLS)."""
        tmp = []
        tmp.append(self._calc_matrix_moment_coeffs())
        if "species_coeffs" in self.optimized:
            tmp.append(self._calc_matrix_species_coeffs())
        return np.hstack(tmp)

    def _calc_matrix_moment_coeffs(self) -> np.ndarray:
        loss = self.loss
        mtp_data = loss.mtp_data
        images = loss.images
        setting = loss.setting
        basis_values = np.array([atoms.calc.engine.basis_values for atoms in images])
        basis_dbdris = np.vstack([atoms.calc.engine.basis_dbdris.T for atoms in images])
        basis_dbdris = basis_dbdris.reshape((-1, mtp_data["alpha_scalar_moments"]))
        tmp = []
        if "energy" in self.minimized:
            tmp.append(np.sqrt(setting["energy-weight"]) * basis_values)
        if "forces" in self.minimized:
            tmp.append(np.sqrt(setting["force-weight"]) * basis_dbdris)
        if "stress" in self.minimized:
            tmp.append(np.sqrt(setting["stress-weight"]) * self._calc_matrix_stress())
        return np.vstack(tmp)

    def _calc_matrix_stress(self) -> np.ndarray:
        images = self.loss.images
        idcs_str = self.loss.idcs_str
        matrix = np.array([images[i].calc.engine.basis_dbdeps.T for i in idcs_str])
        if self.loss.setting["stress-times-volume"]:
            matrix = (matrix.T * self.loss.volumes[idcs_str]).T
        return matrix.reshape((-1, self.loss.mtp_data["alpha_scalar_moments"]))
