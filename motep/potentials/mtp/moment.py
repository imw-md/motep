"""Module for moment tensors."""

import numpy as np
import numpy.typing as npt

from motep.potentials.mtp.data import MTPData
from motep.radial import ChebyshevArrayRadialBasis


class MomentBasis:
    """`MomentBasis`."""

    def __init__(self, mtp_data: MTPData) -> None:
        """Initialize `MomentBasis`."""
        self.mtp_data = mtp_data

    def calculate(
        self,
        itype: int,
        jtypes: list[int],
        r_ijs: npt.NDArray[np.float64],  # (3, neighbors)
        r_abs: npt.NDArray[np.float64],  # (neighbors)
        rb: ChebyshevArrayRadialBasis,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        r"""Calculate basis functions and their derivatives.

        Parameters
        ----------
        r_ijs : np.ndarray
            :math:`\mathbf{r}_j - \mathbf{r}_i`,
            where i is the center atom, and j are the neighboring atoms.

        Returns
        -------
        basis_vals : np.ndarray (alpha_moments_count)
            Values of the basis functions.
        basis_ders : np.ndarray (alpha_moments_count, 3, number_of_atoms)
            Derivatives of the basis functions with respect to :math:`x_j, y_j, z_j`.

        """
        alpha_moments_count = self.mtp_data["alpha_moments_count"]
        alpha_index_basic = self.mtp_data["alpha_index_basic"]
        alpha_index_times = self.mtp_data["alpha_index_times"]
        alpha_moment_mapping = self.mtp_data["alpha_moment_mapping"]

        r_ijs_unit = r_ijs / r_abs
        moment_components = np.zeros(alpha_moments_count)
        moment_jacobian = np.zeros((alpha_moments_count, *r_ijs.shape))  # dEi/dxj

        # Precompute powers
        max_pow = np.max(alpha_index_basic)
        r_unit_pows = _calc_r_unit_pows(r_ijs_unit, max_pow + 1)

        # Compute basic moments
        mu, xpow, ypow, zpow = alpha_index_basic.T
        k = xpow + ypow + zpow

        # `mult0.shape == (alpha_index_basic_count, neighbors)`
        mult0 = r_unit_pows[xpow, 0] * r_unit_pows[ypow, 1] * r_unit_pows[zpow, 2]

        # `val.shape == (alpha_index_basis_count, radial_basis_size, neighbors)`
        val = rb.basis_vs.T * mult0[:, None, :]

        # `der.shape == (alpha_index_basis_count, radial_basis_size, 3, neighbors)`
        der = (rb.basis_ds.T * mult0[:, None, :])[..., None, :] * r_ijs_unit

        der -= (val.T * k).T[:, :, None, :] * r_ijs_unit / r_abs
        der[:, :, 0, :] += (
            rb.basis_vs.T[None, :, :]
            * xpow[:, None, None]
            * r_unit_pows[xpow - 1, None, 0, :]
            * r_unit_pows[ypow, None, 1, :]
            * r_unit_pows[zpow, None, 2, :]
        ) / r_abs
        der[:, :, 1, :] += (
            rb.basis_vs.T[None, :, :]
            * ypow[:, None, None]
            * r_unit_pows[xpow, None, 0, :]
            * r_unit_pows[ypow - 1, None, 1, :]
            * r_unit_pows[zpow, None, 2, :]
        ) / r_abs
        der[:, :, 2, :] += (
            rb.basis_vs.T[None, :, :]
            * zpow[:, None, None]
            * r_unit_pows[xpow, None, 0, :]
            * r_unit_pows[ypow, None, 1, :]
            * r_unit_pows[zpow - 1, None, 2, :]
        ) / r_abs

        # `(alpha_index_basis_count, radial_basis_size, neighbors)`
        coeffs = rb.coeffs[itype, jtypes][:, mu].transpose(1, 2, 0)

        moment_components[: mu.size] = (coeffs * val).sum(axis=(1, 2))
        moment_jacobian[: mu.size] = (coeffs[:, :, None, :] * der).sum(axis=1)

        _contract_moments(moment_components, moment_jacobian, alpha_index_times)

        return (
            moment_components[alpha_moment_mapping],
            moment_jacobian[alpha_moment_mapping],
        )


def _calc_r_unit_pows(r_unit: np.ndarray, max_pow: int) -> np.ndarray:
    r_unit_pows = np.empty((max_pow, *r_unit.shape))
    r_unit_pows[0] = 1.0
    r_unit_pows[1:] = r_unit
    np.multiply.accumulate(r_unit_pows[1:], out=r_unit_pows[1:])
    return r_unit_pows


def _contract_moments(
    moment_components: npt.NDArray[np.float64],
    moment_jacobian: npt.NDArray[np.float64],
    alpha_index_times: npt.NDArray[np.int64],
) -> None:
    """Compute contractions of moments."""
    for ait in alpha_index_times:
        i1, i2, mult, i3 = ait
        moment_components[i3] += mult * moment_components[i1] * moment_components[i2]
        moment_jacobian[i3] += mult * moment_jacobian[i1] * moment_components[i2]
        moment_jacobian[i3] += mult * moment_components[i1] * moment_jacobian[i2]
