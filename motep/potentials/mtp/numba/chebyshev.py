"""Radial basis functions based on Chebyshev polynomials."""

import numba as nb
import numpy as np
import numpy.typing as npt


@nb.njit(nb.float64[:](nb.float64, nb.int64, nb.float64, nb.float64))
def _nb_chebyshev_ene_only(
    r: np.float64,
    number_of_terms: np.int64,
    min_dist: np.float64,
    max_dist: np.float64,
) -> npt.NDArray[np.float64]:
    values = np.empty(number_of_terms)
    r_scaled = (2.0 * r - (min_dist + max_dist)) / (max_dist - min_dist)
    values[0] = 1.0
    values[1] = r_scaled
    for i in range(2, number_of_terms):
        values[i] = 2.0 * r_scaled * values[i - 1] - values[i - 2]
    return values


@nb.njit(
    nb.types.Tuple((nb.float64[:], nb.float64[:]))(
        nb.float64,
        nb.int64,
        nb.float64,
        nb.float64,
    ),
)
def _nb_chebyshev(
    r: np.float64,
    number_of_terms: np.int64,
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


@nb.njit
def _nb_calc_radial_basis_ene_only(
    r_abs: npt.NDArray[np.float64],
    itype: np.int64,
    jtypes: np.ndarray,
    radial_coeffs: np.ndarray,
    scaling: np.float64,
    min_dist: np.float64,
    max_dist: np.float64,
) -> npt.NDArray[np.float64]:
    number_of_neighbors = r_abs.size
    _, _, nmu, rb_size = radial_coeffs.shape
    values = np.zeros((nmu, number_of_neighbors))
    for j in range(number_of_neighbors):
        is_within_cutoff = r_abs[j] < max_dist
        if is_within_cutoff:
            smoothing = (max_dist - r_abs[j]) ** 2
            rb_values = _nb_chebyshev_ene_only(r_abs[j], rb_size, min_dist, max_dist)
            for i_mu in range(nmu):
                for i_rb in range(rb_size):
                    coeffs = scaling * radial_coeffs[itype, jtypes[j], i_mu, i_rb]
                    values[i_mu, j] += coeffs * rb_values[i_rb] * smoothing
    return values


@nb.njit(
    nb.types.Tuple((nb.float64[:, :], nb.float64[:, :]))(
        nb.float64[:],
        nb.int64,
        nb.float64,
        nb.float64,
        nb.float64,
    ),
)
def _nb_calc_radial_basis(
    r_abs: npt.NDArray[np.float64],
    radial_basis_size: np.int64,
    scaling: np.float64,
    min_dist: np.float64,
    max_dist: np.float64,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculate radial basis values."""
    number_of_neighbors = r_abs.size
    values = np.zeros((radial_basis_size, number_of_neighbors))
    derivs = np.zeros((radial_basis_size, number_of_neighbors))
    for j in range(number_of_neighbors):
        if r_abs[j] < max_dist:
            smooth_value = scaling * (max_dist - r_abs[j]) ** 2
            smooth_deriv = -2.0 * scaling * (max_dist - r_abs[j])
            vs0, ds0 = _nb_chebyshev(r_abs[j], radial_basis_size, min_dist, max_dist)
            for k in range(radial_basis_size):
                values[k, j] = vs0[k] * smooth_value
                derivs[k, j] = ds0[k] * smooth_value + vs0[k] * smooth_deriv
    return values, derivs


@nb.njit
def _nb_calc_radial_funcs(
    r_abs: npt.NDArray[np.float64],
    itype: np.int64,
    jtypes: npt.NDArray[np.int64],
    radial_coeffs: np.ndarray,
    scaling: np.float64,
    min_dist: np.float64,
    max_dist: np.float64,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculate radial parts."""
    _, _, radial_funcs_count, radial_basis_size = radial_coeffs.shape
    values, derivs = _nb_calc_radial_basis(
        r_abs,
        radial_basis_size,
        scaling,
        min_dist,
        max_dist,
    )
    radial_part_vs = np.zeros((radial_funcs_count, r_abs.size))
    radial_part_ds = np.zeros((radial_funcs_count, r_abs.size))
    for j in range(r_abs.size):
        for i_mu in range(radial_funcs_count):
            for i_rb in range(radial_basis_size):
                c = radial_coeffs[itype, jtypes[j], i_mu, i_rb]
                radial_part_vs[i_mu, j] += c * values[i_rb, j]
                radial_part_ds[i_mu, j] += c * derivs[i_rb, j]
    return radial_part_vs, radial_part_ds
