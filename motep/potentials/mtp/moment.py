"""Module for moment tensors."""

import numpy as np
import numpy.typing as npt


def _calc_r_unit_pows(r_unit: np.ndarray, max_pow: int) -> np.ndarray:
    r_unit_pows = np.empty((max_pow, *r_unit.shape))
    r_unit_pows[0] = 1.0
    r_unit_pows[1:] = r_unit
    np.multiply.accumulate(r_unit_pows[1:], out=r_unit_pows[1:])
    return r_unit_pows


def calc_moment_basis(
    r_ijs: npt.NDArray[np.float64],  # (3, neighbors)
    r_abs: npt.NDArray[np.float64],  # (neighbors)
    rb_values: npt.NDArray[np.float64],  # (mu, neighbors)
    rb_derivs: npt.NDArray[np.float64],  # (mu, neighbors)
    alpha_moments_count: int,
    alpha_index_basic: npt.NDArray[np.int64],
    alpha_index_times: npt.NDArray[np.int64],
    alpha_moment_mapping: npt.NDArray[np.int64],
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
    r_ijs_unit = r_ijs / r_abs
    moment_components = np.zeros(alpha_moments_count)
    moment_jacobian = np.zeros((alpha_moments_count, *r_ijs.shape))  # dEi/dxj
    # Precompute powers
    max_pow = np.max(alpha_index_basic)
    r_unit_pows = _calc_r_unit_pows(r_ijs_unit, max_pow + 1)

    # Compute basic moments
    mu, xpow, ypow, zpow = alpha_index_basic.T
    k = xpow + ypow + zpow
    mult0 = r_unit_pows[xpow, 0] * r_unit_pows[ypow, 1] * r_unit_pows[zpow, 2]
    val = rb_values[mu] * mult0
    der = (rb_derivs[mu] * mult0)[:, None, :] * r_ijs_unit
    der -= (val * k[:, None])[:, None, :] * r_ijs_unit / r_abs
    der[:, 0] += (
        rb_values[mu]
        * (xpow[:, None] * r_unit_pows[xpow - 1, 0])
        * r_unit_pows[ypow, 1]
        * r_unit_pows[zpow, 2]
    ) / r_abs
    der[:, 1] += (
        rb_values[mu]
        * r_unit_pows[xpow, 0]
        * (ypow[:, None] * r_unit_pows[ypow - 1, 1])
        * r_unit_pows[zpow, 2]
    ) / r_abs
    der[:, 2] += (
        rb_values[mu]
        * r_unit_pows[xpow, 0]
        * r_unit_pows[ypow, 1]
        * (zpow[:, None] * r_unit_pows[zpow - 1, 2])
    ) / r_abs
    moment_components[: mu.size] = val.sum(axis=-1)
    moment_jacobian[: mu.size] = der

    _contract_moments(moment_components, moment_jacobian, alpha_index_times)

    return (
        moment_components[alpha_moment_mapping],
        moment_jacobian[alpha_moment_mapping],
    )


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
