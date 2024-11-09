"""Optimizer for Level 2 MTP."""

from math import sqrt

import numpy as np

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

    @property
    def optimized_default(self) -> list[str]:
        return ["species_coeffs", "radial_coeffs"]

    @property
    def optimized_allowed(self) -> list[str]:
        return ["species_coeffs", "radial_coeffs"]

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
        # Calculate basis functions of `loss.images`
        self.loss(parameters)

        callback = Callback(self.loss)

        # Print the value of the loss function.
        callback(parameters)

        # Update self.data based on the initialized parameters
        self.loss.mtp_data.parameters = parameters

        matrix = self._calc_matrix()
        vector = self._calc_vector()

        coeffs, *_ = np.linalg.lstsq(matrix, vector, rcond=None)

        # Update `mtp_data` and `parameters`.
        parameters = self._update_parameters(coeffs)

        # Print the value of the loss function.
        callback(parameters)

        return parameters

    def _update_parameters(self, coeffs: np.ndarray) -> np.ndarray:
        mtp_data = self.loss.mtp_data
        species_count = mtp_data["species_count"]
        rbs = mtp_data["radial_basis_size"]
        size = species_count * species_count * rbs
        shape = species_count, species_count, rbs

        mtp_data["scaling"] = 1.0
        mtp_data["moment_coeffs"][...] = 0.0
        mtp_data["moment_coeffs"][000] = 1.0
        mtp_data["radial_coeffs"][000] = 0.0
        mtp_data["radial_coeffs"][:, :, 0, :] = coeffs[:size].reshape(shape)
        if "species_coeffs" in self.optimized:
            mtp_data["species_coeffs"] = coeffs[size:]

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
        species_count = mtp_data["species_count"]
        radial_basis_size = mtp_data["radial_basis_size"]
        size = species_count * species_count * radial_basis_size
        matrix = np.stack([atoms.calc.engine.rbd.values for atoms in images])
        if self.loss.setting.energy_per_atom:
            matrix *= self.loss.inverse_numbers_of_atoms[:, None, None, None]
        if self.loss.setting.energy_per_conf:
            matrix /= sqrt(len(images))
        return matrix.reshape(-1, size)

    def _calc_matrix_forces(self) -> np.ndarray:
        species_count = self.loss.mtp_data["species_count"]
        radial_basis_size = self.loss.mtp_data["radial_basis_size"]
        size = species_count * species_count * radial_basis_size

        if not self.loss.idcs_frc.size:
            return np.empty((0, size))

        images = self.loss.images
        idcs = self.loss.idcs_frc

        if self.loss.setting.forces_per_atom:
            matrix = np.stack(
                [
                    images[i].calc.engine.rbd.dqdris.transpose(3, 4, 2, 1, 0)
                    * sqrt(self.loss.inverse_numbers_of_atoms[i])
                    for i in idcs
                ],
            )
        else:
            matrix = np.stack(
                [
                    images[i].calc.engine.rbd.dqdris.transpose(3, 4, 2, 1, 0)
                    for i in idcs
                ],
            )
        if self.loss.setting.forces_per_conf:
            matrix /= sqrt(len(images))
        return matrix.reshape(-1, size)

    def _calc_matrix_stress(self) -> np.ndarray:
        images = self.loss.images
        idcs = self.loss.idcs_str

        species_count = self.loss.mtp_data["species_count"]
        radial_basis_size = self.loss.mtp_data["radial_basis_size"]
        size = species_count * species_count * radial_basis_size

        matrix = np.array([images[i].calc.engine.rbd.dqdeps.T for i in idcs])
        if self.loss.setting.stress_times_volume:
            matrix = (matrix.T * self.loss.volumes[idcs]).T
        return matrix.reshape((-1, size))
