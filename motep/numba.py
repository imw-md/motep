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
    # TODO: precompute distances and send in indices.
    # See also jax implementation of full tensor version
    max_number_of_js = 0
    all_js_list = []
    all_r_ijs = []
    for i in range(number_of_atoms):
        js, r_ijs = engine._get_distances(atoms, i)
        all_js_list.append(js)
        all_r_ijs.append(r_ijs)
        (number_of_js,) = js.shape
        if number_of_js > max_number_of_js:
            max_number_of_js = number_of_js
    shape = (max_number_of_js, number_of_atoms)
    all_js = np.zeros(shape, dtype=int)
    for i in range(number_of_atoms):
        js = all_js_list[i]
        (number_of_js,) = js.shape
        all_js[:number_of_js, i] = js

    energy = 0
    stress = np.zeros((3, 3))
    gradient = np.zeros((number_of_atoms, max_number_of_js, 3))
    for i in range(number_of_atoms):
        js = all_js[:, i]
        r_ijs = all_r_ijs[i]
        (_, number_of_js) = r_ijs.shape
        itype = engine.parameters["species"][atoms.numbers[i]]
        jtypes = np.array([engine.parameters["species"][atoms.numbers[j]] for j in js])
        r_abs = np.linalg.norm(r_ijs, axis=0)
        rb_values, rb_derivs = _nb_calc_radial_basis(
            r_abs, itype, jtypes, radial_coeffs, scaling, min_dist, max_dist
        )
        local_energy, local_gradient = _nb_calc_local_energy_and_gradient(
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
        )
        energy += local_energy
        stress += r_ijs @ local_gradient.T
        gradient[i, :number_of_js, :] = local_gradient.T

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
#         r_abs = np.linalg.norm(r_ijs, axis=0)
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


@nb.njit
def _nb_forces_from_gradient(gradient, all_js, number_of_atoms, max_number_of_js):
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
    r_abs, itype, jtypes, radial_coeffs, scaling, min_dist, max_dist
):
    (nrs,) = r_abs.shape
    _, _, nmu, rb_size = radial_coeffs.shape
    values = np.zeros((nmu, nrs))
    derivs = np.zeros((nmu, nrs))
    for j in range(nrs):
        is_within_cutoff = r_abs[j] < max_dist
        # The below is for the "full_array" implementation
        # is_within_cutoff = 0 < r_abs[j] and r_abs[j] < max_dist
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
    max_pow = int(np.max(alpha_index_basic))
    moment_components = np.zeros(alpha_moments_count)
    moment_jacobian = np.empty((3, number_of_js, alpha_moments_count))
    # Precompute unit vectors and its powers
    r_unit = np.empty((3, number_of_js))
    for j in range(number_of_js):
        for k in range(3):
            r_unit[k, j] = r_ijs[k, j] / r_abs[j]
    r_unit_pows = np.ones((3, number_of_js, max_pow + 1))
    for pow in range(1, max_pow + 1):
        for j in range(number_of_js):
            for k in range(3):
                r_unit_pows[k, j, pow] = r_unit_pows[k, j, pow - 1] * r_unit[k, j]
    # Compute basic moment components and its jacobian wrt r_ijs
    for j in range(number_of_js):
        for aib_i, aib in enumerate(alpha_index_basic):
            mu, xpow, ypow, zpow = aib
            pow = xpow + ypow + zpow
            val = (
                rb_values[mu, j]
                * r_unit_pows[0, j, xpow]
                * r_unit_pows[1, j, ypow]
                * r_unit_pows[2, j, zpow]
            )
            moment_components[aib_i] += val

            for k in range(3):
                moment_jacobian[k, j, aib_i] = (
                    r_unit[k, j]
                    * r_unit_pows[0, j, xpow]
                    * r_unit_pows[1, j, ypow]
                    * r_unit_pows[2, j, zpow]
                    * (rb_derivs[mu, j] - pow * rb_values[mu, j] / r_abs[j])
                )
                if k == 0:
                    moment_jacobian[k, j, aib_i] += (
                        rb_values[mu, j]
                        * (xpow * r_unit_pows[0, j, xpow - 1])
                        * r_unit_pows[1, j, ypow]
                        * r_unit_pows[2, j, zpow]
                        / r_abs[j]
                    )
                elif k == 1:
                    moment_jacobian[k, j, aib_i] += (
                        rb_values[mu, j]
                        * r_unit_pows[0, j, xpow]
                        * (ypow * r_unit_pows[1, j, ypow - 1])
                        * r_unit_pows[2, j, zpow]
                        / r_abs[j]
                    )
                elif k == 2:
                    moment_jacobian[k, j, aib_i] += (
                        rb_values[mu, j]
                        * r_unit_pows[0, j, xpow]
                        * r_unit_pows[1, j, ypow]
                        * (zpow * r_unit_pows[2, j, zpow - 1])
                        / r_abs[j]
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

    # Same for moment jacobian and gradient:
    # # contraction
    # deriv = np.empty((3, nmoments, nrs))
    # for ait in alpha_index_times:
    #     i1, i2, mult, i3 = ait
    #     for j in range(nrs):
    #         for k in range(3):
    #             moment_jacobian[k, j, i3] += mult * (
    #                 moment_jacobian[k, j, i1] * moment_components[i2]
    #                 + moment_components[i1] * moment_jacobian[k, j, i2]
    #             )
    # # basis deriv
    # for basis_i, moment_i in enumerate(alpha_moment_mapping):
    #     for j in range(nrs):
    #         for k in range(3):
    #             deriv[k, basis_i, j] = moment_jacobian[k, j, moment_i]

    # ene_derivs = np.zeros((nrs, 3))
    # for k in range(3):
    #     for j in range(nrs):
    #         ene_derivs[j, k] += moment_coeffs[moment_i] * deriv[k, moment_i, j]

    # alternatively with backpropagation: (saves in the order of 20% for higher levels)
    tmp_moment_ders = np.zeros((alpha_moments_count))
    gradient = np.zeros((3, number_of_js))
    for basis_i, moment_i in enumerate(alpha_moment_mapping):
        tmp_moment_ders[moment_i] = moment_coeffs[basis_i]
    for ait in alpha_index_times[::-1]:
        i1, i2, mult, i3 = ait
        tmp_moment_ders[i2] += tmp_moment_ders[i3] * mult * moment_components[i1]
        tmp_moment_ders[i1] += tmp_moment_ders[i3] * mult * moment_components[i2]
    for aib_i in range(alpha_index_basic.shape[0]):
        for j in range(number_of_js):
            for k in range(3):
                gradient[k, j] += tmp_moment_ders[aib_i] * moment_jacobian[k, j, aib_i]

    return energy, gradient


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
        itype = engine.parameters["species"][atoms.numbers[i]]
        jtypes = np.array([engine.parameters["species"][atoms.numbers[j]] for j in js])
        r_abs = np.linalg.norm(r_ijs, axis=0)
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
