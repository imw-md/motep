"""Module for the optimizer based on linear least squares (LLS)."""

import logging
from abc import abstractmethod
from math import sqrt
from typing import Any

import numpy as np
from ase.stress import voigt_6_to_full_3x3_stress
from scipy.optimize._optimize import OptimizeResult

from motep.loss import LossFunctionBase
from motep.optimizers.base import OptimizerBase
from motep.optimizers.scipy import Callback
from motep.potentials.mtp import get_types

logger = logging.getLogger(__name__)


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
            minimized = ["energy", "forces", "stress"]
        self.minimized = minimized

    @abstractmethod
    def _calc_matrix(self) -> np.ndarray:
        """Calculate the matrix for linear least squares (LLS)."""

    def _calc_matrix_species_coeffs(self) -> np.ndarray:
        loss = self.loss
        images = loss.images
        species = loss.mtp_data.species
        setting = loss.setting
        tmp = []
        if "energy" in self.minimized:
            v = self._calc_matrix_energies_species_coeffs()
            tmp.append(np.sqrt(setting.energy_weight) * v)
        # `species_coeffs` do not affect forces and stresses,
        # and therefore the corresponding sub-mattrices should be zero-filled.
        if "forces" in self.minimized:
            nforces = sum(atoms.calc.targets["forces"].size for atoms in images)
            shape = (nforces, len(species))
            tmp.append(np.zeros(shape))
        if "stress" in self.minimized:
            shape = (9 * len(images), len(species))
            tmp.append(np.zeros(shape))
        return np.vstack(tmp)

    def _calc_matrix_energies_species_coeffs(self) -> np.ndarray:
        loss = self.loss
        species = loss.mtp_data.species
        images = loss.images
        counts = np.full((len(images), len(species)), np.nan)
        for i, atoms in enumerate(images):
            for j, s in enumerate(species):
                counts[i, j] = list(atoms.numbers).count(s)
        if self.loss.setting.energy_per_atom:
            counts *= self.loss.loss_energy.inverse_numbers_of_atoms[:, None]
        if self.loss.setting.energy_per_conf:
            counts /= sqrt(len(images))
        return counts

    def _calc_vector(self) -> np.ndarray:
        """Calculate the vector for linear least squares (LLS)."""
        setting = self.loss.setting
        tmp = []
        if "energy" in self.minimized:
            tmp.append(np.sqrt(setting.energy_weight) * self._calc_vector_energy())
        if "forces" in self.minimized:
            tmp.append(np.sqrt(setting.forces_weight) * self._calc_vector_forces())
        if "stress" in self.minimized:
            tmp.append(np.sqrt(setting.stress_weight) * self._calc_vector_stress())
        return np.hstack(tmp)

    def _calc_vector_energy(self) -> np.ndarray:
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
        species: list[int] = mtp_data.species

        energies = self._calc_target_energies()
        if "species_coeffs" not in self.optimized:
            iterable = (
                np.add.reduce(mtp_data.species_coeffs[get_types(atoms, species)])
                for atoms in images
            )
            energies -= np.fromiter(iterable, dtype=float, count=len(images))
        if self.loss.setting.energy_per_atom:
            energies *= self.loss.loss_energy.inverse_numbers_of_atoms
        if self.loss.setting.energy_per_conf:
            energies /= sqrt(len(images))
        return energies

    def _calc_target_energies(self) -> np.ndarray:
        """Calculate the target energies."""
        images = self.loss.images
        return np.fromiter(
            (atoms.calc.targets["energy"] for atoms in images),
            dtype=float,
            count=len(images),
        )

    def _calc_vector_forces(self) -> np.ndarray:
        if not self.loss.loss_forces.idcs_frc.size:
            return np.empty(0)
        key = "forces"
        images = self.loss.images
        idcs_frc = self.loss.loss_forces.idcs_frc
        if self.loss.setting.forces_per_atom:
            vector = -1.0 * np.hstack(
                [
                    (
                        images[i].calc.targets[key]
                        * sqrt(self.loss.loss_forces.inverse_numbers_of_atoms[i])
                    ).flat
                    for i in idcs_frc
                ],
            )
        else:
            vector = -1.0 * np.hstack(
                [images[i].calc.targets[key].flat for i in idcs_frc],
            )
        if self.loss.setting.forces_per_conf:
            vector /= sqrt(len(images))
        return vector

    def _calc_vector_stress(self) -> np.ndarray:
        key = "stress"
        images = self.loss.images
        idcs_str = self.loss.loss_stress.idcs_str
        f = voigt_6_to_full_3x3_stress
        stresses = np.array([f(images[i].calc.targets[key]) for i in idcs_str])
        if self.loss.setting.stress_times_volume:
            stresses = (stresses.T * self.loss.loss_stress.volumes[idcs_str]).T
        if self.loss.setting.stress_per_conf:
            stresses /= sqrt(len(images))
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

    def optimize(self, **kwargs: dict[str, Any]) -> None:
        parameters = self.loss.mtp_data.parameters
        callback = Callback(self.loss)

        # Calculate basis functions of `loss.images`
        loss_value = self.loss(parameters)
        self.loss.broadcast()

        # Print the value of the loss function.
        callback(OptimizeResult(x=parameters, fun=loss_value))

        # Prepare and solve the LLS problem
        if self.comm.Get_rank() == 0:
            logger.debug("Calculate `matrix`")
            matrix = self._calc_matrix()
            logger.debug("Calculate `vector`")
            vector = self._calc_vector()
            logger.debug("Calculate `coeffs`")
            coeffs = np.linalg.lstsq(matrix, vector, rcond=None)[0]
        else:
            coeffs = None
        coeffs = self.comm.bcast(coeffs, root=0)

        # Update `mtp_data` and `parameters`
        parameters = self._update_parameters(coeffs)

        # Evaluate loss with the new parameters
        loss_value = self.loss(parameters)
        self.loss.broadcast()

        # Print the value of the loss function.
        callback(OptimizeResult(x=parameters, fun=loss_value))

    def _update_parameters(self, coeffs: np.ndarray) -> np.ndarray:
        mtp_data = self.loss.mtp_data
        asm = mtp_data.alpha_scalar_moments
        mtp_data.moment_coeffs = coeffs[:asm]
        if "species_coeffs" in self.optimized:
            mtp_data.species_coeffs = coeffs[asm:]

        return mtp_data.parameters

    def _calc_matrix(self) -> np.ndarray:
        """Calculate the matrix for linear least squares (LLS)."""
        tmp = []
        tmp.append(self._calc_matrix_moment_coeffs())
        if "species_coeffs" in self.optimized:
            tmp.append(self._calc_matrix_species_coeffs())
        return np.hstack(tmp)

    def _calc_matrix_moment_coeffs(self) -> np.ndarray:
        setting = self.loss.setting
        tmp = []
        if "energy" in self.minimized:
            tmp.append(np.sqrt(setting.energy_weight) * self._calc_matrix_energy())
        if "forces" in self.minimized:
            tmp.append(np.sqrt(setting.forces_weight) * self._calc_matrix_forces())
        if "stress" in self.minimized:
            tmp.append(np.sqrt(setting.stress_weight) * self._calc_matrix_stress())
        return np.vstack(tmp)

    def _calc_matrix_energy(self) -> np.ndarray:
        images = self.loss.images
        matrix = np.array([atoms.calc.engine.mbd.values for atoms in images])
        if self.loss.setting.energy_per_atom:
            matrix *= self.loss.loss_energy.inverse_numbers_of_atoms[:, None]
        if self.loss.setting.energy_per_conf:
            matrix /= sqrt(len(images))
        return matrix

    def _calc_matrix_forces(self) -> np.ndarray:
        if not self.loss.loss_forces.idcs_frc.size:
            return np.empty((0, self.loss.mtp_data.alpha_scalar_moments))
        images = self.loss.images
        idcs_frc = self.loss.loss_forces.idcs_frc
        if self.loss.setting.forces_per_atom:
            matrix = np.vstack(
                [
                    images[i].calc.engine.mbd.dbdris.transpose(1, 2, 0)
                    * sqrt(self.loss.loss_forces.inverse_numbers_of_atoms[i])
                    for i in idcs_frc
                ],
            )
        else:
            matrix = np.vstack(
                [images[i].calc.engine.mbd.dbdris.transpose(1, 2, 0) for i in idcs_frc],
            )
        if self.loss.setting.forces_per_conf:
            matrix /= sqrt(len(images))
        return matrix.reshape((-1, self.loss.mtp_data.alpha_scalar_moments))

    def _calc_matrix_stress(self) -> np.ndarray:
        images = self.loss.images
        idcs_str = self.loss.loss_stress.idcs_str
        matrix = np.array([images[i].calc.engine.mbd.dbdeps.T for i in idcs_str])
        if self.loss.setting.stress_times_volume:
            matrix = (matrix.T * self.loss.loss_stress.volumes[idcs_str]).T
        if self.loss.setting.stress_per_conf:
            matrix /= sqrt(len(images))
        return matrix.reshape((-1, self.loss.mtp_data.alpha_scalar_moments))
