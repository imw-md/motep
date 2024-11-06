from typing import Any

import numba as nb
import numpy as np


@nb.njit
def _nb_linalg_norm(r_ijs: np.ndarray) -> np.ndarray:
    return np.sqrt((r_ijs**2).sum(axis=1))


#
# The below implementation is from some reason slightly slower up to level 10 for larger systems.
# This should be tested again for higher levels
#
# def numba_calc_energy_and_forces__full_arrays(
#     engine,
#     atoms,
#     alpha_moments_count,
#     alpha_moment_mapping,
#     alpha_index_basic,
#     alpha_index_times,
#     scaling,
#     min_dist,
#     max_dist,
#     species_coeffs,
#     moment_coeffs,
#     radial_coeffs,
# ):
#     # TODO: take out engine from here and precompute distances and send in indices.
#     # See also jax implementation of full tensor version
#     # Something like this:
#     #####################
#     # positions = atoms.positions
#     # offsets = engine.precomputed_offsets
#     # indices = np.array(
#     #     [engine._neighbor_list.get_neighbors(i)[0] for i in range(number_of_atoms)]
#     # )
#     # all_distances = (positions - positions[indices.T]).transpose(
#     #     1, 2, 0
#     # ) - offsets.transpose(0, 2, 1)
#     #####################
#
#     assert len(alpha_index_times.shape) == 2
#     number_of_atoms = len(atoms)
#     max_number_of_js = 0
#     for i in range(number_of_atoms):
#         js, r_ijs = engine._get_distances(atoms, i)
#         (number_of_js,) = js.shape
#         if number_of_js > max_number_of_js:
#             max_number_of_js = number_of_js
#     shape = (max_number_of_js, number_of_atoms)
#     all_js = np.zeros(shape, dtype=int)
#     all_r_ijs = np.zeros((3,) + shape, dtype=float)
#     all_r_abs = -1 * np.ones(shape, dtype=float)
#     all_itypes = np.empty(number_of_atoms, dtype=int)
#     all_jtypes = np.zeros(shape, dtype=int)
#     for i in range(number_of_atoms):
#         js, r_ijs = engine._get_distances(atoms, i)
#         r_abs = np.sqrt(np.add.reduce(r_ijs**2, axis=0))
#         itype = engine.parameters["species"][atoms.numbers[i]]
#         jtypes = np.array([engine.parameters["species"][atoms.numbers[j]] for j in js])
#         (number_of_js,) = js.shape
#         all_js[:number_of_js, i] = js
#         all_r_ijs[:, :number_of_js, i] = r_ijs
#         all_r_abs[:number_of_js, i] = r_abs
#         all_itypes[i] = itype
#         all_jtypes[:number_of_js, i] = jtypes
#
#     all_js = all_js
#
#     energy, gradient = _nb_calc_energy_and_gradient(
#         all_r_ijs,
#         all_r_abs,
#         all_itypes,
#         all_jtypes,
#         alpha_moments_count,
#         alpha_moment_mapping,
#         alpha_index_basic,
#         alpha_index_times,
#         scaling,
#         min_dist,
#         max_dist,
#         radial_coeffs,
#         species_coeffs,
#         moment_coeffs,
#         number_of_atoms,
#         max_number_of_js,
#     )
#     stress = np.zeros((3, 3))
#     for i in range(number_of_atoms):
#         r_ijs = all_r_ijs[:, :, i]
#         loc_gradient = gradient[i, : r_ijs.shape[1], :]
#         stress += r_ijs @ loc_gradient
#
#     forces = _nb_forces_from_gradient(
#         gradient, all_js, number_of_atoms, max_number_of_js
#     )
#
#     if atoms.cell.rank == 3:
#         stress = (stress + stress.T) * 0.5  # symmetrize
#         stress /= atoms.get_volume()
#         stress = stress.flat[[0, 4, 8, 5, 2, 1]]
#     else:
#         stress = np.full(6, np.nan)
#
#     return energy, forces, stress
#
#
# @nb.njit(fastmath=True)
# def _nb_calc_energy_and_gradient(
#     all_r_ijs,
#     all_r_abs,
#     all_itypes,
#     all_jtypes,
#     alpha_moments_count,
#     alpha_moment_mapping,
#     alpha_index_basic,
#     alpha_index_times,
#     scaling,
#     min_dist,
#     max_dist,
#     radial_coeffs,
#     species_coeffs,
#     moment_coeffs,
#     number_of_atoms,
#     max_number_of_js,
# ):
#     energy = 0
#     gradient = np.zeros((number_of_atoms, max_number_of_js, 3))
#     for i in range(number_of_atoms):
#         rb_values, rb_derivs = _nb_calc_radial_basis(
#             all_r_abs[:, i],
#             all_itypes[i],
#             all_jtypes[:, i],
#             radial_coeffs,
#             scaling,
#             min_dist,
#             max_dist,
#         )
#         loc_energy, loc_gradient = _nb_calc_local_energy_and_gradient(
#             all_r_ijs[:, :, i],
#             all_r_abs[:, i],
#             rb_values,
#             rb_derivs,
#             alpha_moments_count,
#             alpha_moment_mapping,
#             alpha_index_basic,
#             alpha_index_times,
#             all_itypes[i],
#             species_coeffs,
#             moment_coeffs,
#         )
#         energy += loc_energy
#         for j in range(max_number_of_js):
#             for k in range(3):
#                 gradient[i, j, k] = loc_gradient[k, j]
#     return energy, gradient


@nb.njit(
    nb.float64[:, :](
        nb.float64[:, :, :],
        nb.int64[:, :],
        nb.int64,
    ),
)
def _nb_forces_from_gradient(
    gradient: np.ndarray,
    all_js: np.ndarray,
    max_number_of_js: int,
) -> np.ndarray:
    number_of_atoms = gradient.shape[0]
    forces = np.zeros((number_of_atoms, 3))
    for i in range(number_of_atoms):
        for i_j in range(max_number_of_js):
            j = all_js[i_j, i]
            for k in range(3):
                forces[i, k] += gradient[i, i_j, k]
                forces[j, k] -= gradient[i, i_j, k]
    return forces


@nb.njit
def _nb_chebyshev(r, number_of_terms, min_dist, max_dist):
    values = np.empty((number_of_terms))
    derivs = np.empty((number_of_terms))
    r_scaled = (2 * r - (min_dist + max_dist)) / (max_dist - min_dist)
    r_deriv = 2 / (max_dist - min_dist)
    values[0] = 1
    values[1] = r_scaled
    derivs[0] = 0
    derivs[1] = r_deriv
    for i in range(2, number_of_terms):
        values[i] = 2 * r_scaled * values[i - 1] - values[i - 2]
        derivs[i] = (
            2 * (r_scaled * derivs[i - 1] + r_deriv * values[i - 1]) - derivs[i - 2]
        )
    return values, derivs


@nb.njit
def _nb_calc_radial_basis(
    r_abs: np.ndarray[Any, np.float64],
    radial_basis_size: np.int64,
    scaling: np.float64,
    min_dist: np.float64,
    max_dist: np.float64,
) -> tuple[np.ndarray[Any, np.float64], np.ndarray[Any, np.float64]]:
    """Calculate radial basis values."""
    number_of_neighbors = r_abs.size
    values = np.zeros((number_of_neighbors, radial_basis_size))
    derivs = np.zeros((number_of_neighbors, radial_basis_size))
    for j in range(number_of_neighbors):
        if r_abs[j] < max_dist:
            smooth_value = scaling * (max_dist - r_abs[j]) ** 2
            smooth_deriv = -2.0 * scaling * (max_dist - r_abs[j])
            vs0, ds0 = _nb_chebyshev(r_abs[j], radial_basis_size, min_dist, max_dist)
            for k in range(radial_basis_size):
                values[j, k] = vs0[k] * smooth_value
                derivs[j, k] = ds0[k] * smooth_value + vs0[k] * smooth_deriv
    return values, derivs


@nb.njit
def _nb_calc_radial_funcs(
    r_abs: np.ndarray[Any, np.float64],
    itype: np.int64,
    jtypes: np.ndarray[Any, np.int64],
    radial_coeffs: np.ndarray,
    scaling: np.float64,
    min_dist: np.float64,
    max_dist: np.float64,
) -> tuple[np.ndarray[Any, np.float64], np.ndarray[Any, np.float64]]:
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
                radial_part_vs[i_mu, j] += c * values[j, i_rb]
                radial_part_ds[i_mu, j] += c * derivs[j, i_rb]
    return radial_part_vs, radial_part_ds


@nb.njit(nb.float64[:, :](nb.float64[:, :], nb.float64[:]))
def _calc_r_unit(r_ijs: np.ndarray, r_abs: np.ndarray) -> np.ndarray:
    return r_ijs[:, :] / r_abs[:, None]


@nb.njit
def _calc_r_unit_pows(r_unit: np.ndarray, max_pow: int) -> np.ndarray:
    number_of_js = r_unit.shape[0]
    r_unit_pows = np.ones((max_pow + 1, number_of_js, 3))
    for pow in range(1, max_pow + 1):
        for j in range(number_of_js):
            for k in range(3):
                r_unit_pows[pow, j, k] = r_unit_pows[pow - 1, j, k] * r_unit[j, k]
    return r_unit_pows


@nb.njit(
    (
        nb.float64[:],
        nb.float64[:, :],
        nb.int64[:, :],
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64[:],
        nb.float64[:, :, :],
    ),
)
def _calc_moment_basic(
    r_abs,
    r_ijs,
    alpha_index_basic,
    rb_values,
    rb_derivs,
    moment_components,
    moment_jacobian,
) -> None:
    """Compute basic moment components and its jacobian wrt `r_ijs`."""
    # Precompute unit vectors
    r_unit = _calc_r_unit(r_ijs, r_abs)

    # Precompute powers
    max_pow = int(np.max(alpha_index_basic))
    r_unit_pows = _calc_r_unit_pows(r_unit, max_pow)

    number_of_js = moment_jacobian.shape[1]
    for aib_i, aib in enumerate(alpha_index_basic):
        mu, xpow, ypow, zpow = aib
        xyzpow = xpow + ypow + zpow
        for j in range(number_of_js):
            mult0 = 1.0
            mult0 *= r_unit_pows[xpow, j, 0]
            mult0 *= r_unit_pows[ypow, j, 1]
            mult0 *= r_unit_pows[zpow, j, 2]
            moment_components[aib_i] += rb_values[mu, j] * mult0
            for k in range(3):
                moment_jacobian[aib_i, j, k] = (
                    r_unit[j, k]
                    * mult0
                    * (rb_derivs[mu, j] - xyzpow * rb_values[mu, j] / r_abs[j])
                )
                if k == 0:
                    moment_jacobian[aib_i, j, k] += (
                        rb_values[mu, j]
                        * (xpow * r_unit_pows[xpow - 1, j, 0])
                        * r_unit_pows[ypow, j, 1]
                        * r_unit_pows[zpow, j, 2]
                        / r_abs[j]
                    )
                elif k == 1:
                    moment_jacobian[aib_i, j, k] += (
                        rb_values[mu, j]
                        * r_unit_pows[xpow, j, 0]
                        * (ypow * r_unit_pows[ypow - 1, j, 1])
                        * r_unit_pows[zpow, j, 2]
                        / r_abs[j]
                    )
                elif k == 2:
                    moment_jacobian[aib_i, j, k] += (
                        rb_values[mu, j]
                        * r_unit_pows[xpow, j, 0]
                        * r_unit_pows[ypow, j, 1]
                        * (zpow * r_unit_pows[zpow - 1, j, 2])
                        / r_abs[j]
                    )


@nb.njit(
    (
        nb.int64[:, :],
        nb.float64[:],
        nb.float64[:, :, :],
    ),
)
def _calc_moment_times(
    alpha_index_times,
    moment_components,
    moment_jacobian,
):
    number_of_js = moment_jacobian.shape[1]
    for ait in alpha_index_times:
        i1, i2, mult, i3 = ait
        moment_components[i3] += mult * moment_components[i1] * moment_components[i2]
        for j in range(number_of_js):
            for k in range(3):
                moment_jacobian[i3, j, k] += mult * (
                    moment_jacobian[i1, j, k] * moment_components[i2]
                    + moment_components[i1] * moment_jacobian[i2, j, k]
                )


# @nb.njit(
#     nb.float64[:, :](
#         nb.int64[:, :],
#         nb.int64[:],
#         nb.float64[:],
#         nb.float64[:],
#         nb.float64[:, :, :],
#     ),
# )
# def _propagate_forward(
#     alpha_index_times,
#     alpha_moment_mapping,
#     moment_coeffs,
#     moment_components,
#     moment_jacobian,
# ):
#     """Calculate gradients using the forward propagation."""
#     _calc_moment_times(alpha_index_times, moment_components, moment_jacobian)
#     _, number_of_js, _ = moment_jacobian.shape
#     gradient = np.zeros((number_of_js, 3))
#     for basis_i, moment_i in enumerate(alpha_moment_mapping):
#         for j in range(number_of_js):
#             for k in range(3):
#                 gradient[j, k] += (
#                     moment_coeffs[basis_i] * moment_jacobian[moment_i, j, k]
#                 )

#     return gradient


@nb.njit(
    nb.float64[:, :](
        nb.int64[:, :],
        nb.int64[:, :],
        nb.int64[:],
        nb.float64[:],
        nb.float64[:],
        nb.float64[:, :, :],
    ),
)
def _propagate_backward(
    alpha_index_basic,
    alpha_index_times,
    alpha_moment_mapping,
    moment_coeffs,
    moment_components,
    moment_jacobian,
):
    # alternatively with backpropagation: (saves in the order of 20% for higher levels)
    alpha_moments_count, number_of_js, _ = moment_jacobian.shape
    tmp_moment_ders = np.zeros((alpha_moments_count))
    for basis_i, moment_i in enumerate(alpha_moment_mapping):
        tmp_moment_ders[moment_i] = moment_coeffs[basis_i]
    for ait in alpha_index_times[::-1]:
        i1, i2, mult, i3 = ait
        tmp_moment_ders[i2] += tmp_moment_ders[i3] * mult * moment_components[i1]
        tmp_moment_ders[i1] += tmp_moment_ders[i3] * mult * moment_components[i2]

    gradient = np.zeros((number_of_js, 3))
    for aib_i in range(alpha_index_basic.shape[0]):
        for j in range(number_of_js):
            for k in range(3):
                gradient[j, k] += tmp_moment_ders[aib_i] * moment_jacobian[aib_i, j, k]

    return gradient


@nb.njit(
    nb.types.Tuple((nb.float64, nb.float64[:, :]))(
        nb.float64[:, :],
        nb.float64[:],
        nb.float64[:, :],
        nb.float64[:, :],
        nb.int64,
        nb.int64[:],
        nb.int64[:, :],
        nb.int64[:, :],
        nb.int64,
        nb.float64[:],
        nb.float64[:],
    )
)
def _nb_calc_local_energy_and_gradient(
    r_ijs,
    r_abs,
    rb_values,
    rb_derivs,
    alpha_moments_count,
    alpha_moment_mapping,
    alpha_index_basic,
    alpha_index_times,
    itype,
    species_coeffs,
    moment_coeffs,
):
    (number_of_js,) = r_abs.shape
    moment_components = np.zeros(alpha_moments_count)
    moment_jacobian = np.zeros((alpha_moments_count, number_of_js, 3))

    _calc_moment_basic(
        r_abs,
        r_ijs,
        alpha_index_basic,
        rb_values,
        rb_derivs,
        moment_components,
        moment_jacobian,
    )

    # For moments and energy:
    # Compute moment contraction components
    for ait in alpha_index_times:
        i1, i2, mult, i3 = ait
        moment_components[i3] += mult * moment_components[i1] * moment_components[i2]
    # Extract basis elements and multiply with moment coefficients
    energy = species_coeffs[itype]
    for basis_i, moment_i in enumerate(alpha_moment_mapping):
        energy += moment_coeffs[basis_i] * moment_components[moment_i]

    gradient = _propagate_backward(
        alpha_index_basic,
        alpha_index_times,
        alpha_moment_mapping,
        moment_coeffs,
        moment_components,
        moment_jacobian,
    )

    return energy, gradient


@nb.njit(
    nb.types.Tuple((nb.float64[:], nb.float64[:, :, :]))(
        nb.float64[:],
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64[:, :],
        nb.int64,
        nb.int64[:],
        nb.int64[:, :],
        nb.int64[:, :],
    ),
)
def _nb_calc_moment(
    r_abs: np.ndarray,
    r_ijs: np.ndarray,
    rb_values: np.ndarray,
    rb_derivs: np.ndarray,
    alpha_moments_count: np.int64,
    alpha_moment_mapping: np.ndarray,
    alpha_index_basic: np.ndarray,
    alpha_index_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    (number_of_js,) = r_abs.shape
    moment_components = np.zeros(alpha_moments_count)
    moment_jacobian = np.zeros((alpha_moments_count, number_of_js, 3))

    _calc_moment_basic(
        r_abs,
        r_ijs,
        alpha_index_basic,
        rb_values,
        rb_derivs,
        moment_components,
        moment_jacobian,
    )

    _calc_moment_times(
        alpha_index_times,
        moment_components,
        moment_jacobian,
    )

    return (
        moment_components[alpha_moment_mapping],
        moment_jacobian[alpha_moment_mapping],
    )


#
# Energy-only numba implementation
#
def numba_calc_energy(
    engine,
    atoms,
    alpha_moments_count,
    alpha_moment_mapping,
    alpha_index_basic,
    alpha_index_times,
    scaling,
    min_dist,
    max_dist,
    species_coeffs,
    moment_coeffs,
    radial_coeffs,
):
    assert len(alpha_index_times.shape) == 2
    number_of_atoms = len(atoms)
    max_number_of_js = 0
    for i in range(number_of_atoms):
        js, r_ijs = engine._get_distances(atoms, i)
        (number_of_js,) = js.shape
        if number_of_js > max_number_of_js:
            max_number_of_js = number_of_js
    shape = (max_number_of_js, number_of_atoms)
    all_js = np.zeros(shape, dtype=int)
    all_r_ijs = []
    for i in range(number_of_atoms):
        js, r_ijs = engine._get_distances(atoms, i)
        all_r_ijs.append(r_ijs)
        (number_of_js,) = js.shape
        all_js[:number_of_js, i] = js

    energy = 0
    for i in range(number_of_atoms):
        js = all_js[:, i]
        r_ijs = all_r_ijs[i]
        (_, number_of_js) = r_ijs.shape
        itype = engine.mtp_data["species"][atoms.numbers[i]]
        jtypes = np.array([engine.mtp_data["species"][atoms.numbers[j]] for j in js])
        r_abs = np.sqrt(np.add.reduce(r_ijs**2, axis=0))
        rb_values = _nb_calc_radial_basis_ene_only(
            r_abs, itype, jtypes, radial_coeffs, scaling, min_dist, max_dist
        )
        loc_energy = _nb_calc_local_energy_only(
            r_ijs,
            r_abs,
            rb_values,
            alpha_moments_count,
            alpha_moment_mapping,
            alpha_index_basic,
            alpha_index_times,
            itype,
            species_coeffs,
            moment_coeffs,
        )
        energy += loc_energy
    return energy


@nb.njit
def _nb_calc_radial_basis_ene_only(
    r_abs, itype, jtypes, radial_coeffs, scaling, min_dist, max_dist
):
    (nrs,) = r_abs.shape
    _, _, nmu, rb_size = radial_coeffs.shape
    values = np.zeros((nmu, nrs))
    for j in range(nrs):
        is_within_cutoff = r_abs[j] < max_dist
        if is_within_cutoff:
            smoothing = (max_dist - r_abs[j]) ** 2
            rb_values = _nb_chebyshev_ene_only(r_abs[j], rb_size, min_dist, max_dist)
            for i_mu in range(nmu):
                for i_rb in range(rb_size):
                    coeffs = scaling * radial_coeffs[itype, jtypes[j], i_mu, i_rb]
                    values[i_mu, j] += coeffs * rb_values[i_rb] * smoothing
    return values


@nb.njit
def _nb_chebyshev_ene_only(r, number_of_terms, min_dist, max_dist):
    values = np.empty((number_of_terms))
    r_scaled = (2 * r - (min_dist + max_dist)) / (max_dist - min_dist)
    values[0] = 1
    values[1] = r_scaled
    for i in range(2, number_of_terms):
        values[i] = 2 * r_scaled * values[i - 1] - values[i - 2]
    return values


@nb.njit(
    nb.float64(
        nb.float64[:, :],
        nb.float64[:],
        nb.float64[:, :],
        nb.int64,
        nb.int64[:],
        nb.int64[:, :],
        nb.int64[:, :],
        nb.int64,
        nb.float64[:],
        nb.float64[:],
    )
)
def _nb_calc_local_energy_only(
    r_ijs,
    r_abs,
    rb_values,
    alpha_moments_count,
    alpha_moment_mapping,
    alpha_index_basic,
    alpha_index_times,
    itype,
    species_coeffs,
    moment_coeffs,
):
    nrs = r_abs.shape[0]
    max_pow = int(np.max(alpha_index_basic))
    r_ijs_unit = r_ijs / r_abs
    moment_components = np.zeros(alpha_moments_count)
    # Precompute powers
    r_ijs_pows = np.ones((3, nrs, max_pow + 1))
    for pow in range(1, max_pow + 1):
        for j in range(nrs):
            for k in range(3):
                r_ijs_pows[k, j, pow] = r_ijs_pows[k, j, pow - 1] * r_ijs_unit[k, j]
    # Compute basic moment components
    for i, aib in enumerate(alpha_index_basic):
        for j in range(nrs):
            mu, xpow, ypow, zpow = aib
            val = (
                rb_values[mu, j]
                * r_ijs_pows[0, j, xpow]
                * r_ijs_pows[1, j, ypow]
                * r_ijs_pows[2, j, zpow]
            )
            moment_components[i] += val
    # Compute moment contraction components
    for ait in alpha_index_times:
        i1, i2, mult, i3 = ait
        moment_components[i3] += mult * moment_components[i1] * moment_components[i2]
    # Extract basis elements and multiply with moment coefficients
    energy = species_coeffs[itype]
    for basis_i, moment_i in enumerate(alpha_moment_mapping):
        energy += moment_coeffs[basis_i] * moment_components[moment_i]
    return energy
