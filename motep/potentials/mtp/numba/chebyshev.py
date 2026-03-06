"""Radial basis functions based on Chebyshev polynomials."""

import numba as nb
import numpy as np
import numpy.typing as npt


@nb.njit(
    nb.types.Tuple((nb.float64[:], nb.float64[:]))(
        nb.float64,
        nb.int32,
        nb.float64,
        nb.float64,
    ),
)
def _nb_chebyshev(
    r: np.float64,
    number_of_terms: np.int32,
    min_dist: np.float64,
    max_dist: np.float64,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    values = np.empty(number_of_terms)
    derivs = np.empty(number_of_terms)
    r_scaled = (2.0 * r - (min_dist + max_dist)) / (max_dist - min_dist)
    r_deriv = 2.0 / (max_dist - min_dist)
    values[0] = 1.0
    values[1] = r_scaled
    derivs[0] = 0.0
    derivs[1] = r_deriv
    for i in range(2, number_of_terms):
        values[i] = 2.0 * r_scaled * values[i - 1] - values[i - 2]
        derivs[i] = (
            2.0 * (r_scaled * derivs[i - 1] + r_deriv * values[i - 1]) - derivs[i - 2]
        )
    return values, derivs


@nb.njit(
    nb.types.Tuple((nb.float64[:, :], nb.float64[:, :]))(
        nb.float64[:],
        nb.int32,
        nb.float64,
        nb.float64,
        nb.float64,
    ),
)
def calc_radial_basis(
    r_abs: npt.NDArray[np.float64],
    radial_basis_size: np.int32,
    scaling: np.float64,
    min_dist: np.float64,
    max_dist: np.float64,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculate radial basis values."""
    values = np.zeros((radial_basis_size, r_abs.size))
    derivs = np.zeros((radial_basis_size, r_abs.size))
    for j in range(r_abs.size):
        if r_abs[j] < max_dist:
            smooth_value = scaling * (max_dist - r_abs[j]) ** 2
            smooth_deriv = -2.0 * scaling * (max_dist - r_abs[j])
            vs0, ds0 = _nb_chebyshev(r_abs[j], radial_basis_size, min_dist, max_dist)
            for i_rb in range(radial_basis_size):
                values[i_rb, j] = vs0[i_rb] * smooth_value
                derivs[i_rb, j] = ds0[i_rb] * smooth_value + vs0[i_rb] * smooth_deriv
    return values, derivs


@nb.njit(
    nb.float64[:, :](
        nb.int32,
        nb.int32[:],
        nb.float64[:, :],
        nb.float64[:, :, :, :],
    ),
)
def sum_radial_terms(
    itype: np.int32,
    jtypes: npt.NDArray[np.int32],
    basis: npt.NDArray[np.float64],
    coeffs: np.ndarray,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Sum radial terms."""
    njs = basis.shape[1]
    _, _, rfc, rbs = coeffs.shape
    sum_ = np.zeros((rfc, njs))
    for j in range(njs):
        for mu in range(rfc):
            for i_rb in range(rbs):
                c = coeffs[itype, jtypes[j], mu, i_rb]
                sum_[mu, j] += c * basis[i_rb, j]
    return sum_
