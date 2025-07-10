"""Jax implementation of MTP."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from ase import Atoms

from motep.potentials.mtp import get_types
from motep.potentials.mtp.base import EngineBase
from motep.potentials.mtp.data import MTPData

from .conversion import BasisConverter, moments_count_to_level_map
from .moment import MomentBasis

jax.config.update("jax_enable_x64", True)  # noqa: FBT003


class JaxMTPEngine(EngineBase):
    """MTP Engine in 'full tensor' version based on jax."""

    def __init__(self, *args, **kwargs) -> None:
        """Intialize the engine."""
        self.moment_basis = None
        self.basis_converter = None
        super().__init__(*args, **kwargs)

    def update(self, mtp_data: MTPData) -> None:
        """Update MTP parameters.

        Raises
        ------
        ValueError: If `level` is updated after initialization.

        """
        super().update(mtp_data)
        if self.mtp_data.alpha_moments_count is not None:
            level = moments_count_to_level_map[mtp_data.alpha_moments_count]
            if self.moment_basis is None:
                self.moment_basis = MomentBasis(level)
                self.moment_basis.init_moment_mappings()
                self.basis_converter = BasisConverter(self.moment_basis)
            elif self.moment_basis.max_level != level:
                msg = "Changing level is not allowed, use a new instance instead."
                raise ValueError(msg)
            self.basis_converter.remap_mlip_moment_coeffs(self.mtp_data)

    def _calculate(self, atoms: Atoms) -> tuple:
        mtp_data = self.mtp_data
        itypes = jnp.array(get_types(atoms, mtp_data.species))
        all_js, all_rijs = [jnp.array(_) for _ in self._get_all_distances(atoms)]
        all_jtypes = itypes[all_js]

        energies, gradients = _calc_local_energy_and_derivs(
            all_rijs,
            itypes,
            all_jtypes,
            mtp_data.species_coeffs,
            self.basis_converter.remapped_coeffs,
            mtp_data.radial_coeffs,
            mtp_data.scaling,
            mtp_data.min_dist,
            mtp_data.max_dist,
            # Static parameters:
            mtp_data.radial_coeffs.shape[3],
            self.moment_basis.basic_moments,
            self.moment_basis.pair_contractions,
            self.moment_basis.scalar_contractions,
        )

        forces = np.array(gradients.sum(axis=1))
        np.subtract.at(forces, all_js, gradients)

        stress = np.array((all_rijs.transpose((0, 2, 1)) @ gradients).sum(axis=0))

        return energies, forces, stress


@partial(jax.jit, static_argnums=(9, 10, 11, 12))
@partial(jax.vmap, in_axes=(0,) * 3 + (None,) * 10, out_axes=0)
def _calc_local_energy_and_derivs(
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
) -> tuple:
    energy = _calc_local_energy(
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
    derivs = jax.jacobian(_calc_local_energy)(
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
def _calc_local_energy(
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
) -> jnp.array:
    r_abs = jnp.linalg.norm(r_ijs, axis=1)
    smoothing = jnp.where(r_abs < max_dist, (max_dist - r_abs) ** 2, 0)
    radial_basis = _chebyshev_basis(r_abs, min_dist, max_dist, rb_size)
    # j, rb_size
    rb_values = (
        scaling
        * smoothing
        * jnp.einsum("jmn, jn -> mj", radial_coeffs[itype, jtypes], radial_basis)
    )
    basis = _calc_basis(
        r_ijs, r_abs, rb_values, basic_moments, pair_contractions, scalar_contractions
    )
    return species_coeffs[itype] + jnp.dot(moment_coeffs, basis)


@partial(jax.jit, static_argnums=(3, 4, 5))
def _calc_basis(
    r_ijs,
    r_abs,
    rb_values,
    # Static parameters:
    basic_moments,
    pair_contractions,
    scalar_contractions,
):
    calculated_moments = _calc_moments(r_ijs, r_abs, rb_values, basic_moments)
    for contraction in pair_contractions:
        m1 = calculated_moments[contraction[0]]
        m2 = calculated_moments[contraction[1]]
        calculated_moments[contraction] = _contract_over_axes(m1, m2, contraction[3])
    basis = []
    for contraction in scalar_contractions:
        b = calculated_moments[contraction]
        basis.append(b)
    return jnp.array(basis)


@partial(jax.jit, static_argnums=(3,))
@partial(jax.vmap, in_axes=(0, None, None, None), out_axes=0)
def _chebyshev_basis(
    r: jnp.array,
    min_dist: float,
    max_dist: float,
    size: int,
) -> jnp.array:
    r_scaled = (2 * r - (min_dist + max_dist)) / (max_dist - min_dist)
    rb = [1, r_scaled]
    for i in range(2, size):
        rb.append(2 * r_scaled * rb[i - 1] - rb[i - 2])
    return jnp.array(rb)


@partial(jax.jit, static_argnums=(3,))
def _calc_moments(
    r_ijs: jnp.array,
    r_abs: jnp.array,
    rb_values: jnp.array,
    moments: tuple,
) -> jnp.array:
    calculated_moments = {}
    r_ijs_unit = (r_ijs.T / r_abs).T
    for moment in moments:
        mu = moment[0]
        nu = moment[1]
        m = _make_tensor(r_ijs_unit, nu)
        m = (m.T * rb_values[mu]).sum(axis=-1)
        calculated_moments[moment] = m
    return calculated_moments


@partial(jax.vmap, in_axes=(0, None), out_axes=0)
def _make_tensor(r: jnp.array, nu: int) -> jnp.array:
    m = 1
    for _ in range(nu):
        m = jnp.tensordot(r, m, axes=0)
    return m


@partial(jax.jit, static_argnums=(2,))
def _contract_over_axes(m1: jnp.array, m2: jnp.array, axes: tuple) -> jnp.array:
    return jnp.tensordot(m1, m2, axes=axes)
