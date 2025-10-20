"""Optimizer for Level 2 MTP."""

import logging
from math import sqrt
from typing import Any

import numpy as np
from scipy.optimize._optimize import OptimizeResult

from motep.optimizers.lls import LLSOptimizerBase
from motep.optimizers.scipy import Callback

logger = logging.getLogger(__name__)


class Level2MTPOptimizer(LLSOptimizerBase):
    """Optimizer for Level 2 MTP.

    Attributes
    ----------
    optimized : list[str]
        Coefficients to be optimized.
        The elements must be some of `species_coeffs` and `radial_coeffs`.

    """

    @property
    def optimized_default(self) -> list[str]:
        return ["species_coeffs", "radial_coeffs"]

    @property
    def optimized_allowed(self) -> list[str]:
        return ["species_coeffs", "radial_coeffs"]

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

        # Update `mtp_data` and `parameters`.
        parameters = self._update_parameters(coeffs)

        # Evaluate loss with the new parameters
        loss_value = self.loss(parameters)
        self.loss.broadcast()

        # Print the value of the loss function.
        callback(OptimizeResult(x=parameters, fun=loss_value))

    def _update_parameters(self, coeffs: np.ndarray) -> np.ndarray:
        mtp_data = self.loss.mtp_data
        species_count = mtp_data.species_count
        rbs = mtp_data.radial_basis_size
        size = species_count * species_count * rbs
        shape = species_count, species_count, rbs

        mtp_data.scaling = 1.0
        mtp_data.moment_coeffs[...] = 0.0
        mtp_data.moment_coeffs[000] = 1.0
        mtp_data.radial_coeffs[000] = 0.0
        mtp_data.radial_coeffs[...] = coeffs[:size].reshape(shape)[:, :, None, :]
        if "species_coeffs" in self.optimized:
            mtp_data.species_coeffs = coeffs[size:]

        return mtp_data.parameters

    def _calc_matrix(self) -> np.ndarray:
        """Calculate the matrix for linear least squares (LLS)."""
        tmp = []
        tmp.append(self._calc_matrix_radial_coeffs())
        if "species_coeffs" in self.optimized:
            tmp.append(self._calc_matrix_species_coeffs())
        return np.hstack(tmp)

    def _calc_matrix_radial_coeffs(self) -> np.ndarray:
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
        loss = self.loss
        mtp_data = loss.mtp_data
        images = loss.images
        species_count = mtp_data.species_count
        radial_basis_size = mtp_data.radial_basis_size
        size = species_count * species_count * radial_basis_size
        matrix = np.stack([atoms.calc.engine.rbd.values for atoms in images])
        if self.loss.setting.energy_per_atom:
            cs = self.loss.loss_energy.inverse_numbers_of_atoms
            matrix *= cs[:, None, None, None]
        if self.loss.setting.energy_per_conf:
            matrix /= sqrt(len(images))
        return matrix.reshape(-1, size)

    def _calc_matrix_forces(self) -> np.ndarray:
        species_count = self.loss.mtp_data.species_count
        radial_basis_size = self.loss.mtp_data.radial_basis_size
        size = species_count * species_count * radial_basis_size

        if not self.loss.loss_forces.idcs_frc.size:
            return np.empty((0, size))

        images = self.loss.images
        idcs = self.loss.loss_forces.idcs_frc

        if self.loss.setting.forces_per_atom:
            matrix = np.hstack(
                [
                    (
                        images[i].calc.engine.rbd.dqdris.transpose(3, 4, 0, 1, 2)
                        * sqrt(self.loss.loss_energy.inverse_numbers_of_atoms[i])
                    ).flat
                    for i in idcs
                ],
            )
        else:
            matrix = np.hstack(
                [
                    images[i].calc.engine.rbd.dqdris.transpose(3, 4, 0, 1, 2).flat
                    for i in idcs
                ],
            )
        if self.loss.setting.forces_per_conf:
            matrix /= sqrt(len(images))
        return matrix.reshape(-1, size)

    def _calc_matrix_stress(self) -> np.ndarray:
        images = self.loss.images
        idcs = self.loss.loss_stress.idcs_str

        species_count = self.loss.mtp_data.species_count
        radial_basis_size = self.loss.mtp_data.radial_basis_size
        size = species_count * species_count * radial_basis_size

        matrix = np.array([images[i].calc.engine.rbd.dqdeps.T for i in idcs])
        if self.loss.setting.stress_times_volume:
            matrix = (matrix.T * self.loss.loss_stress.volumes[idcs]).T
        if self.loss.setting.stress_per_conf:
            matrix /= sqrt(len(images))
        return matrix.reshape((-1, size))
