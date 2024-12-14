from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from motep.potentials.mtp import get_types

jax.config.update("jax_enable_x64", True)


# @partial(jax.jit, static_argnums=(9, 10, 11, 12))
def calc_energy_forces_stress(
    engine,
    atoms,
    species,
    scaling,
    min_dist,
    max_dist,
    species_coeffs,
    moment_coeffs,
    radial_coeffs,
    # Static parameters:
    basic_moments,
    pair_contractions,
    scalar_contractions,
):
    itypes = jnp.array(get_types(atoms, species))
    all_js, all_rijs = engine._get_all_distances(atoms)
    all_js, all_rijs = jnp.array(all_js), jnp.array(all_rijs)
    all_jtypes = itypes[all_js]

    local_energy, local_gradient = _jax_calc_local_energy_and_derivs(
        all_rijs,
        itypes,
        all_jtypes,
        species_coeffs,
        moment_coeffs,
        radial_coeffs,
        scaling,
        min_dist,
        max_dist,
        # Static parameters:
        radial_coeffs.shape[3],
        basic_moments,
        pair_contractions,
        scalar_contractions,
    )

    energy = local_energy.sum()

    forces = np.array(local_gradient.sum(axis=1))
    np.subtract.at(forces, all_js, local_gradient)

    stress = np.array((all_rijs.transpose((0, 2, 1)) @ local_gradient).sum(axis=0))
    # ijk @ ikj -> ikk -> kk
    if atoms.cell.rank == 3:
        stress = (stress + stress.T) * 0.5  # symmetrize
        stress /= atoms.get_volume()
        stress = stress.flat[[0, 4, 8, 5, 2, 1]]
    else:
        stress = np.full(6, np.nan)

    return energy, forces, stress


@partial(jax.jit, static_argnums=(9, 10, 11, 12))
@partial(jax.vmap, in_axes=(0,) * 3 + (None,) * 10, out_axes=0)
def _jax_calc_local_energy_and_derivs(
    r_ijs,
    itype,
    jtypes,
    species_coeffs,
    moment_coeffs,
    radial_coeffs,
    scaling,
    min_dist,
    max_dist,
    # Static parameters:
    rb_size,
    basic_moments,
    pair_contractions,
    scalar_contractions,
):
    energy = _jax_calc_local_energy(
        r_ijs,
        itype,
        jtypes,
        species_coeffs,
        moment_coeffs,
        radial_coeffs,
        scaling,
        min_dist,
        max_dist,
        # Static parameters:
        rb_size,
        basic_moments,
        pair_contractions,
        scalar_contractions,
    )
    derivs = jax.jacobian(_jax_calc_local_energy)(
        r_ijs,
        itype,
        jtypes,
        species_coeffs,
        moment_coeffs,
        radial_coeffs,
        scaling,
        min_dist,
        max_dist,
        # Static parameters:
        rb_size,
        basic_moments,
        pair_contractions,
        scalar_contractions,
    )
    return energy, derivs


@partial(jax.jit, static_argnums=(9, 10, 11, 12))
def _jax_calc_local_energy(
    r_ijs,
    itype,
    jtypes,
    species_coeffs,
    moment_coeffs,
    radial_coeffs,
    scaling,
    min_dist,
    max_dist,
    # Static parameters:
    rb_size,
    basic_moments,
    pair_contractions,
    scalar_contractions,
):
    r_abs = jnp.linalg.norm(r_ijs, axis=1)
    smoothing = jnp.where(r_abs < max_dist, (max_dist - r_abs) ** 2, 0)
    radial_basis = _jax_chebyshev_basis(r_abs, rb_size, min_dist, max_dist)
    # j, rb_size
    rb_values = (
        scaling
        * smoothing
        * jnp.einsum("jmn, jn -> mj", radial_coeffs[itype, jtypes], radial_basis)
    )
    basis = _jax_calc_basis(
        r_ijs, r_abs, rb_values, basic_moments, pair_contractions, scalar_contractions
    )
    energy = species_coeffs[itype] + jnp.dot(moment_coeffs, basis)
    return energy


@partial(jax.jit, static_argnums=(3, 4, 5))
def _jax_calc_basis(
    r_ijs,
    r_abs,
    rb_values,
    # Static parameters:
    basic_moments,
    pair_contractions,
    scalar_contractions,
):
    calculated_moments = _jax_calc_moments(r_ijs, r_abs, rb_values, basic_moments)
    for contraction in pair_contractions:
        m1 = calculated_moments[contraction[0]]
        m2 = calculated_moments[contraction[1]]
        calculated_moments[contraction] = _jax_contract_over_axes(
            m1, m2, contraction[3]
        )
    basis = []
    for contraction in scalar_contractions:
        b = calculated_moments[contraction]
        basis.append(b)
    return jnp.array(basis)


@partial(jax.jit, static_argnums=[1, 2, 3])
@partial(jax.vmap, in_axes=[0, None, None, None], out_axes=0)
def _jax_chebyshev_basis(r, number_of_terms, min_dist, max_dist):
    r_scaled = (2 * r - (min_dist + max_dist)) / (max_dist - min_dist)
    rb = [1, r_scaled]
    for i in range(2, number_of_terms):
        rb.append(2 * r_scaled * rb[i - 1] - rb[i - 2])
    return jnp.array(rb)


@partial(jax.jit, static_argnums=(3,))
def _jax_calc_moments(r_ijs, r_abs, rb_values, moments):
    calculated_moments = {}
    r_ijs_unit = (r_ijs.T / r_abs).T
    for moment in moments:
        mu = moment[0]
        nu = moment[1]
        m = _jax_make_tensor(r_ijs_unit, nu)
        m = (m.T * rb_values[mu]).sum(axis=-1)
        calculated_moments[moment] = m
    return calculated_moments


@partial(jax.vmap, in_axes=[0, None], out_axes=0)
def _jax_make_tensor(r, nu):
    m = 1
    for _ in range(nu):
        m = jnp.tensordot(r, m, axes=0)
    return m


@partial(jax.jit, static_argnums=(2,))
def _jax_contract_over_axes(m1, m2, axes):
    calculated_contraction = jnp.tensordot(m1, m2, axes=axes)
    return calculated_contraction
