"""Module for moment tensors."""

import numpy as np
import numpy.typing as npt

from motep.potentials.mtp.data import MTPData
from motep.potentials.mtp.numpy.chebyshev import ChebyshevArrayRadialBasis


class MomentBasis:
    """`MomentBasis`."""

    def __init__(self, mtp_data: MTPData) -> None:
        """Initialize `MomentBasis`."""
        self.mtp_data = mtp_data

    def calculate(
        self,
        itype: int,
        jtypes: list[int],
        r_ijs: npt.NDArray[np.float64],  # (neighbors, 3)
        r_abs: npt.NDArray[np.float64],  # (neighbors)
        rb: ChebyshevArrayRadialBasis,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        r"""Calculate basis functions and their derivatives.

        Parameters
        ----------
        r_ijs : np.ndarray (number_of_neighbors, 3)
            :math:`\mathbf{r}_j - \mathbf{r}_i`,
            where i is the center atom, and j are the neighboring atoms.

        Returns
        -------
        basis_vals : np.ndarray (alpha_moments_count)
            Values of the basis functions.
        basis_ders : np.ndarray (alpha_moments_count, number_of_atoms, 3)
            Derivatives of the basis functions with respect to :math:`x_j, y_j, z_j`.

        """
        species_count = self.mtp_data.species_count
        rfs = self.mtp_data.radial_funcs_count
        rbs = self.mtp_data.radial_basis_size
        amc = self.mtp_data.alpha_moments_count
        alpha_index_basic = self.mtp_data.alpha_index_basic
        alpha_index_times = self.mtp_data.alpha_index_times
        alpha_moment_mapping = self.mtp_data.alpha_moment_mapping

        r_ijs_unit = (r_ijs.T / r_abs).T
        moment_values = np.zeros(amc)
        moment_jac_rs = np.zeros((amc, *r_ijs.shape))  # dEi/dxj
        moment_jac_cs = np.zeros((amc, species_count, rfs, rbs))
        moment_jac_rc = np.zeros((amc, species_count, rfs, rbs, *r_ijs.shape))

        # Precompute powers
        max_pow = np.max(alpha_index_basic)
        r_unit_pows = _calc_r_unit_pows(r_ijs_unit, max_pow + 1)

        # Compute basic moments
        mu, xpow, ypow, zpow = alpha_index_basic.T
        k = xpow + ypow + zpow

        # `mult0.shape == (alpha_index_basic_count, neighbors)`
        mult0 = (
            r_unit_pows[xpow, :, 0] * r_unit_pows[ypow, :, 1] * r_unit_pows[zpow, :, 2]
        )

        # f * tensor = dMb/dc (before summation over neighbors)
        # `val.shape == (alpha_index_basis_count, radial_basis_size, neighbors)`
        val = rb.basis_vs * mult0[:, None, :]

        # d(d(f * tensor)/dr)/dc = d(dMb/dr)/dc (before summation over neighbors)
        # `der.shape == (alpha_index_basis_count, radial_basis_size, neighbors, 3)`
        der = (rb.basis_ds * mult0[:, None, :])[..., None] * r_ijs_unit

        der -= (val.T * k).T[..., None] * r_ijs_unit / r_abs[:, None]
        der[..., 0] += (
            rb.basis_vs[None, :, :]
            * xpow[:, None, None]
            * r_unit_pows[xpow - 1, None, :, 0]
            * r_unit_pows[ypow, None, :, 1]
            * r_unit_pows[zpow, None, :, 2]
        ) / r_abs
        der[..., 1] += (
            rb.basis_vs[None, :, :]
            * ypow[:, None, None]
            * r_unit_pows[xpow, None, :, 0]
            * r_unit_pows[ypow - 1, None, :, 1]
            * r_unit_pows[zpow, None, :, 2]
        ) / r_abs
        der[..., 2] += (
            rb.basis_vs[None, :, :]
            * zpow[:, None, None]
            * r_unit_pows[xpow, None, :, 0]
            * r_unit_pows[ypow, None, :, 1]
            * r_unit_pows[zpow - 1, None, :, 2]
        ) / r_abs

        # `(alpha_index_basis_count, radial_basis_size, neighbors)`
        coeffs = rb.coeffs[itype, jtypes][:, mu].transpose(1, 2, 0)

        moment_values[: mu.size] = (coeffs * val).sum(axis=(1, 2))
        moment_jac_rs[: mu.size] = (coeffs[:, :, :, None] * der).sum(axis=1)
        ijs = np.arange(len(jtypes))
        for imu, _ in enumerate(mu):
            np.add.at(moment_jac_cs[imu, :, _], jtypes, val.T[:, :, imu])
            np.add.at(
                moment_jac_rc.transpose(0, 2, 1, 4, 3, 5)[imu, _],
                (jtypes, ijs),
                der.transpose(0, 2, 1, 3)[imu],
            )

        _contract_moments(alpha_index_times, moment_values, moment_jac_rs)

        moment_coeffs = self.mtp_data.moment_coeffs

        dedcs, dgdcs = calc_dedcs_and_dgdcs(
            self.mtp_data.alpha_index_basic_count,
            alpha_index_times,
            alpha_moment_mapping,
            moment_coeffs,
            moment_values,
            moment_jac_rs,
            moment_jac_cs,
            moment_jac_rc,
        )

        return (
            moment_values[alpha_moment_mapping],
            moment_jac_rs[alpha_moment_mapping],
            dedcs,
            dgdcs,
        )


def _calc_r_unit_pows(r_unit: np.ndarray, max_pow: int) -> np.ndarray:
    r_unit_pows = np.empty((max_pow, *r_unit.shape))
    r_unit_pows[0] = 1.0
    r_unit_pows[1:] = r_unit
    np.multiply.accumulate(r_unit_pows[1:], out=r_unit_pows[1:])
    return r_unit_pows


def _contract_moments(
    alpha_index_times: npt.NDArray[np.int64],
    moment_values: npt.NDArray[np.float64],
    moment_jac_rs: npt.NDArray[np.float64],
) -> None:
    """Compute contractions of moments."""
    for ait in alpha_index_times:
        i1, i2, mult, i3 = ait
        moment_values[i3] += mult * moment_values[i1] * moment_values[i2]

        moment_jac_rs[i3] += mult * (
            moment_jac_rs[i1] * moment_values[i2]
            + moment_values[i1] * moment_jac_rs[i2]
        )


def calc_dedcs_and_dgdcs(
    alpha_index_basic_count: np.int64,
    alpha_index_times: np.ndarray,
    alpha_moment_mapping: np.ndarray,
    moment_coeffs: np.ndarray,
    moment_values: np.ndarray,
    moment_jac_rs: np.ndarray,
    moment_jac_cs: np.ndarray,
    moment_jac_rc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate dV/dc and d(dV/dr)/dc.

    - dV/dc: Jacobian of site energy to radial basis coefficients
    - d(dV/dr)/dc: Jacobians of site energy gradients to radial basis coefficients

    Returns
    -------
    dedcs : np.ndarray
        dV/dc.
    dgdcs : np.ndarray
        d(dV/dr)/dc.

    """
    aibc = alpha_index_basic_count
    dedmb = np.zeros_like(moment_values)
    dgdmb = np.zeros_like(moment_jac_rs)
    dedmb[alpha_moment_mapping] = moment_coeffs  # dV/dB
    for ait in alpha_index_times[::-1]:
        i1, i2, mult, i3 = ait
        dedmb[i1] += mult * dedmb[i3] * moment_values[i2]
        dedmb[i2] += mult * dedmb[i3] * moment_values[i1]
    for ait in alpha_index_times:
        i1, i2, mult, i3 = ait
        dgdmb[i1] += mult * dedmb[i3] * moment_jac_rs[i2]
        dgdmb[i2] += mult * dedmb[i3] * moment_jac_rs[i1]
    for ait in alpha_index_times[::-1]:
        i1, i2, mult, i3 = ait
        dgdmb[i1] += mult * dgdmb[i3] * moment_values[i2]
        dgdmb[i2] += mult * dgdmb[i3] * moment_values[i1]
    dedcs = (moment_jac_cs[:aibc].T @ dedmb[:aibc]).T
    dgdcs = (moment_jac_rc[:aibc].T @ dedmb[:aibc]).T
    dgdcs += (
        moment_jac_cs[:aibc, ..., None, None] * dgdmb[:aibc, None, None, None]
    ).sum(axis=0)
    return dedcs, dgdcs
