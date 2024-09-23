import numba as nb
import numpy as np


def numba_calc_energy_and_forces(
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
    # TODO: take out engine from here and precompute distances and send in indices.
    # See also jax implementation of full tensor version
    max_number_of_js = 0
    for i in range(number_of_atoms):
        js, r_ijs = engine._get_distances(atoms, i)
        (number_of_js,) = js.shape
        if number_of_js > max_number_of_js:
            max_number_of_js = number_of_js
    shape = (max_number_of_js, number_of_atoms)
    all_js = np.zeros(shape, dtype=int)

    all_r_ijs = []
    max_njs = 0
    for i in range(number_of_atoms):
        js, r_ijs = engine._get_distances(atoms, i)
        if len(js) > max_njs:
            max_njs = len(js)
        all_r_ijs.append(r_ijs)
        (number_of_js,) = js.shape
        all_js[:number_of_js, i] = js

    energy = 0
    stress = np.zeros((3, 3))
    gradient = np.zeros((number_of_atoms, max_njs, 3))
    for i in range(number_of_atoms):
        js = all_js[:, i]
        r_ijs = all_r_ijs[i]
        itype = engine.parameters["species"][atoms.numbers[i]]
        jtypes = np.array([engine.parameters["species"][atoms.numbers[j]] for j in js])
        r_abs = np.linalg.norm(r_ijs, axis=0)
        loc_energy, loc_gradient = _nb_calc_local_energy_and_derivs(
            r_ijs,
            r_abs,
            itype,
            jtypes,
            alpha_moments_count,
            alpha_moment_mapping,
            alpha_index_basic,
            alpha_index_times,
            scaling,
            min_dist,
            max_dist,
            radial_coeffs,
            species_coeffs,
            moment_coeffs,
        )
        energy += loc_energy
        stress += r_ijs @ loc_gradient
        gradient[i, : loc_gradient.shape[0], :] = loc_gradient

    forces = _nb_forces_from_gradient(
        gradient, all_js, number_of_atoms, max_number_of_js
    )

    if atoms.cell.rank == 3:
        stress = (stress + stress.T) * 0.5  # symmetrize
        stress /= atoms.get_volume()
        stress = stress.flat[[0, 4, 8, 5, 2, 1]]
    else:
        stress = np.full(6, np.nan)

    return energy, forces, stress


@nb.njit
def _nb_forces_from_gradient(gradient, all_js, number_of_atoms, max_number_of_js):
    forces = np.zeros((number_of_atoms, 3))
    for i in range(number_of_atoms):
        for i_j in range(max_number_of_js):
            j = all_js[i_j, i]
            for k in range(3):
                forces[i, k] -= gradient[i, i_j, k]
                forces[j, k] += gradient[i, i_j, k]
    return forces


@nb.njit
def _nb_calc_local_energy_and_derivs(
    r_ijs,
    r_abs,
    itype,
    jtypes,
    alpha_moments_count,
    alpha_moment_mapping,
    alpha_index_basic,
    alpha_index_times,
    scaling,
    min_dist,
    max_dist,
    radial_coeffs,
    species_coeffs,
    moment_coeffs,
):
    rb_values, rb_derivs = _nb_calc_radial_basis_and_deriv(
        r_abs, itype, jtypes, radial_coeffs, scaling, min_dist, max_dist
    )
    basis, bderiv = _nb_calc_moment_basis_and_deriv(
        r_ijs,
        r_abs,
        rb_values,
        rb_derivs,
        alpha_moments_count,
        alpha_moment_mapping,
        alpha_index_basic,
        alpha_index_times,
    )
    energy, derivs = _nb_convolve_with_coeffs(
        basis, bderiv, itype, species_coeffs, moment_coeffs
    )
    return energy, derivs


@nb.njit
def _nb_convolve_with_coeffs(basis, bderiv, itype, species_coeffs, moment_coeffs):
    nrs = bderiv.shape[2]
    nbasis = bderiv.shape[1]
    energy = species_coeffs[itype]
    derivs = np.zeros((nrs, 3))
    for i in range(nbasis):
        energy += moment_coeffs[i] * basis[i]
        for k in range(3):
            for j in range(nrs):
                derivs[j, k] += moment_coeffs[i] * bderiv[k, i, j]
    return energy, derivs


@nb.njit
def _nb_chebyshev(r, rb_size, min_dist, max_dist):
    rb_values = np.empty((rb_size))
    rb_derivs = np.empty((rb_size))
    r_scaled = (2 * r - (min_dist + max_dist)) / (max_dist - min_dist)
    r_deriv = 2 / (max_dist - min_dist)
    rb_values[0] = 1
    rb_values[1] = r_scaled
    rb_derivs[0] = 0
    rb_derivs[1] = r_deriv
    for i in range(2, rb_size):
        rb_values[i] = 2 * r_scaled * rb_values[i - 1] - rb_values[i - 2]
        rb_derivs[i] = (
            2 * (r_scaled * rb_derivs[i - 1] + r_deriv * rb_values[i - 1])
            - rb_derivs[i - 2]
        )
    return rb_values, rb_derivs


@nb.njit
def _nb_calc_radial_basis_and_deriv(
    r_abs, itype, jtypes, radial_coeffs, scaling, min_dist, max_dist
):
    (nrs,) = r_abs.shape
    _, _, nmu, rb_size = radial_coeffs.shape
    values = np.zeros((nmu, nrs))
    derivs = np.zeros((nmu, nrs))
    for j in range(nrs):
        is_within_cutoff = r_abs[j] < max_dist
        if is_within_cutoff:
            smoothing = (max_dist - r_abs[j]) ** 2
            smooth_deriv = -2 * (max_dist - r_abs[j])
            rb_values, rb_derivs = _nb_chebyshev(r_abs[j], rb_size, min_dist, max_dist)
            for i_mu in range(nmu):
                for i_rb in range(rb_size):
                    coeffs = scaling * radial_coeffs[itype, jtypes[j], i_mu, i_rb]
                    values[i_mu, j] += coeffs * rb_values[i_rb] * smoothing
                    derivs[i_mu, j] += coeffs * (
                        rb_derivs[i_rb] * smoothing + rb_values[i_rb] * smooth_deriv
                    )
    return values, derivs


@nb.njit(
    nb.types.Tuple((nb.float64[:], nb.float64[:, :, :]))(
        nb.float64[:, :],
        nb.float64[:],
        nb.float64[:, :],
        nb.float64[:, :],
        nb.int64,
        nb.int64[:],
        nb.int64[:, :],
        nb.int64[:, :],
    )
)
def _nb_calc_moment_basis_and_deriv(
    r_ijs,
    r_abs,
    rb_values,
    rb_derivs,
    alpha_moments_count,
    alpha_moment_mapping,
    alpha_index_basic,
    alpha_index_times,
):
    (nrs,) = r_abs.shape
    max_pow = int(np.max(alpha_index_basic))
    moment_components = np.zeros(alpha_moments_count)
    moment_jacobian = np.zeros((3, alpha_moments_count, nrs))
    # Precompute powers
    coord_pows = np.ones((max_pow + 1, 3, nrs))
    dist_pows = np.ones((max_pow + 1, nrs))
    for pow in range(1, max_pow + 1):
        for j in range(nrs):
            dist_pows[pow, j] = dist_pows[pow - 1, j] * r_abs[j]
            for k in range(3):
                coord_pows[pow, k, j] = coord_pows[pow - 1, k, j] * r_ijs[k, j]
    # Compute basic moments
    for j in range(nrs):
        for i, aib in enumerate(alpha_index_basic):
            mu, xpow, ypow, zpow = aib
            pow = xpow + ypow + zpow
            val = (
                rb_values[mu, j]
                * coord_pows[xpow, 0, j]
                * coord_pows[ypow, 1, j]
                * coord_pows[zpow, 2, j]
                / dist_pows[pow, j]
            )
            moment_components[i] += val

            der = np.empty(3)
            der[0] = (
                rb_values[mu, j]
                / dist_pows[pow, j]
                * xpow
                * (
                    coord_pows[xpow - 1, 0, j]
                    * coord_pows[ypow, 1, j]
                    * coord_pows[zpow, 2, j]
                )
            )
            der[1] = (
                rb_values[mu, j]
                / dist_pows[pow, j]
                * ypow
                * (
                    coord_pows[xpow, 0, j]
                    * coord_pows[ypow - 1, 1, j]
                    * coord_pows[zpow, 2, j]
                )
            )
            der[2] = (
                rb_values[mu, j]
                / dist_pows[pow, j]
                * zpow
                * (
                    coord_pows[xpow, 0, j]
                    * coord_pows[ypow, 1, j]
                    * coord_pows[zpow - 1, 2, j]
                )
            )
            for k in range(3):
                der[k] += (
                    (
                        rb_derivs[mu, j] / dist_pows[pow, j]
                        - pow * rb_values[mu, j] / dist_pows[pow, j] / r_abs[j]
                    )
                    * (
                        coord_pows[xpow, 0, j]
                        * coord_pows[ypow, 1, j]
                        * coord_pows[zpow, 2, j]
                        / r_abs[j]
                    )
                    * r_ijs[k, j]
                )
                moment_jacobian[k, i, j] += der[k]
    # Compute contractions
    for ait in alpha_index_times:
        i1, i2, mult, i3 = ait
        moment_components[i3] += mult * moment_components[i1] * moment_components[i2]
        # TODO: Test performance of backwards propagation
        for j in range(nrs):
            for k in range(3):
                moment_jacobian[k, i3, j] += mult * (
                    moment_jacobian[k, i1, j] * moment_components[i2]
                    + moment_components[i1] * moment_jacobian[k, i2, j]
                )
    # Compute basis
    nmoments = alpha_moment_mapping.shape[0]
    basis = np.empty(nmoments)
    deriv = np.empty((3, nmoments, nrs))
    for basis_i, moment_i in enumerate(alpha_moment_mapping):
        basis[basis_i] = moment_components[moment_i]
        for k in range(3):
            for j in range(nrs):
                deriv[k, basis_i, j] = moment_jacobian[k, moment_i, j]
    return basis, deriv


#
# Energy only numba implementation for moment basis (not used at the moment)
#
# @nb.njit(
#     nb.float64[:](
#         nb.float64[:, :],
#         nb.float64[:],
#         nb.float64[:, :],
#         nb.int64,
#         nb.int64[:],
#         nb.int64[:, :],
#         nb.int64[:, :],
#     )
# )
# def numba_calc_moment_basis(
#     r_ijs,
#     r_abs,
#     rb_values,
#     alpha_moments_count,
#     alpha_moment_mapping,
#     alpha_index_basic,
#     alpha_index_times,
# ):
#     nrs = r_abs.shape[0]
#     max_pow = int(np.max(alpha_index_basic))
#     r_ijs_unit = r_ijs / r_abs
#     moment_components = np.zeros(alpha_moments_count)
#     # Precompute powers
#     val_pows = np.ones((max_pow + 1, 3, nrs))
#     for pow in range(1, max_pow + 1):
#         for k in range(3):
#             for j in range(nrs):
#                 val_pows[pow, k, j] = val_pows[pow - 1, k, j] * r_ijs_unit[k, j]
#     # Compute basic moments
#     for i, aib in enumerate(alpha_index_basic):
#         for j in range(nrs):
#             mu, xpow, ypow, zpow = aib
#             val = (
#                 rb_values[mu, j]
#                 * val_pows[xpow, 0, j]
#                 * val_pows[ypow, 1, j]
#                 * val_pows[zpow, 2, j]
#             )
#             moment_components[i] += val
#     # Compute contractions
#     for ait in alpha_index_times:
#         i1, i2, mult, i3 = ait
#         moment_components[i3] += mult * moment_components[i1] * moment_components[i2]
#     # Compute basis
#     nmoments = alpha_moment_mapping.shape[0]
#     basis = np.empty(nmoments)
#     for basis_i, moment_i in enumerate(alpha_moment_mapping):
#         basis[basis_i] = moment_components[moment_i]
#     return basis
