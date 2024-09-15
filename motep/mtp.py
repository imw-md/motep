"""MTP writtin in Python.

Original version: Axel Forslund
Modified version: Yuji Ikeda
"""

from typing import Any

import numpy as np
from ase import Atoms
from ase.neighborlist import NewPrimitiveNeighborList
from numpy.polynomial import Chebyshev


def init_radial_basis_functions(
    radial_coeffs: np.ndarray,
    min_dist: float,
    max_dist: float,
) -> np.ndarray:  # array of Chebyshev objects
    radial_basis_funcs = []
    domain = [min_dist, max_dist]
    nspecies, _, nmu, _ = radial_coeffs.shape
    for i0 in range(nspecies):
        for i1 in range(nspecies):
            for i2 in range(nmu):
                p = Chebyshev(radial_coeffs[i0, i1, i2], domain=domain)
                radial_basis_funcs.append(p)
    shape = nspecies, nspecies, nmu
    return np.array(radial_basis_funcs).reshape(shape)


def calc_radial_basis(
    radial_basis_funcs: np.ndarray,  # array of Chebyshev objects
    r_abs: np.ndarray,
    itype: int,
    jtypes: list[int],
    scaling: float,
    min_dist: float,
    max_dist: float,
    radial_funcs_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    is_within_cutoff = (min_dist < r_abs) & (r_abs < max_dist)
    smooth_values = scaling * (max_dist - r_abs) ** 2
    smooth_derivs = -2.0 * scaling * (max_dist - r_abs)
    rb_values = np.zeros((radial_funcs_count, r_abs.size))
    rb_derivs = np.zeros((radial_funcs_count, r_abs.size))
    for mu in range(radial_funcs_count):
        for j, jtype in enumerate(jtypes):
            if is_within_cutoff[j]:
                rb_funcs = radial_basis_funcs[itype, jtype, mu]
                v = rb_funcs(r_abs[j]) * smooth_values[j]
                rb_values[mu, j] = v
                d0 = rb_funcs.deriv()(r_abs[j]) * smooth_values[j]
                d1 = rb_funcs(r_abs[j]) * smooth_derivs[j]
                d = d0 + d1
                rb_derivs[mu, j] = d
    return rb_values, rb_derivs


class MTP:
    """MLIP-2 MTP."""

    def __init__(self, parameters: dict[str, Any]):
        self.parameters = parameters
        if "species" not in self.parameters:
            species = {_: _ for _ in range(self.parameters["species_count"])}
            self.parameters["species"] = species
        self.radial_basis_funcs = init_radial_basis_functions(
            self.parameters["radial_coeffs"],
            self.parameters["min_dist"],
            self.parameters["max_dist"],
        )
        self.results = {}

    def calc_radial_basis(
        self,
        r_abs: np.ndarray,
        itype: int,
        jtypes: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        return calc_radial_basis(
            self.radial_basis_funcs,
            r_abs,
            itype,
            jtypes,
            self.parameters["scaling"],
            self.parameters["min_dist"],
            self.parameters["max_dist"],
            self.parameters["radial_funcs_count"],
        )

    def _get_local_energy(self, atoms: Atoms, i: int):
        itype = self.parameters["species"][atoms.numbers[i]]
        js, r_ijs = self._get_distances(atoms, i)
        jtypes = [self.parameters["species"][atoms.numbers[j]] for j in js]
        r_abs = np.linalg.norm(r_ijs, axis=0)
        rb_values, rb_derivs = self.calc_radial_basis(r_abs, itype, jtypes)
        basis_values = calc_moment_basis(
            r_ijs,
            r_abs,
            rb_values,
            self.parameters["alpha_moments_count"],
            self.parameters["alpha_index_basic"],
            self.parameters["alpha_index_times"],
            self.parameters["alpha_moment_mapping"],
        )
        species_coeffs = self.parameters["species_coeffs"]
        moment_coeffs = self.parameters["moment_coeffs"]
        return species_coeffs[itype] + moment_coeffs @ basis_values

    def get_energy(self, atoms: Atoms):
        self.update_neighbor_list(atoms)
        generator = (self._get_local_energy(atoms, i) for i in range(len(atoms)))
        self.results["energies"] = np.fromiter(generator, dtype=float)
        self.results["energy"] = self.results["energies"].sum()
        return self.results["energy"]

    def _initiate_neighbor_list(self, atoms: Atoms):
        self._neighbor_list = NewPrimitiveNeighborList(
            cutoffs=self.parameters["max_dist"],
            skin=0.3,  # cutoff + skin is used, recalc only if diff in pos > skin
            self_interaction=False,  # Exclude [0, 0, 0]
            bothways=True,  # return both ij and ji
        )
        self._neighbor_list.update(atoms.pbc, atoms.cell, atoms.positions)
        self.precomputed_offsets = _compute_offsets(self._neighbor_list, atoms)

    def update_neighbor_list(self, atoms: Atoms):
        if self._neighbor_list is None:
            self._initiate_neighbor_list(atoms)
        else:
            self._neighbor_list.update(atoms.pbc, atoms.cell, atoms.positions)
            self.precomputed_offsets = _compute_offsets(self._neighbor_list, atoms)

    def _get_distances(
        self,
        atoms: Atoms,
        index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        indices_js, _ = self._neighbor_list.get_neighbors(index)
        offsets = self.precomputed_offsets[index]
        pos_js = atoms.positions[indices_js] + offsets
        dist_vectors = atoms.positions[index] - pos_js
        return indices_js, dist_vectors.T


def _compute_offsets(nl: NewPrimitiveNeighborList, atoms: Atoms):
    cell = atoms.cell
    return np.array([nl.get_neighbors(j)[1] @ cell for j in range(len(atoms))])


def calc_moment_basis(
    r_ijs: np.ndarray,  # (3, neighbors)
    r_abs: np.ndarray,  # (neighbors)
    rb_values: np.ndarray,  # (neighbors, mu)
    alpha_moments_count,
    alpha_index_basic: int,
    alpha_index_times: int,
    alpha_moment_mapping: np.ndarray,
):
    r_ijs_unit = r_ijs / r_abs
    moment_components = np.zeros(alpha_moments_count)
    # Precompute powers
    max_pow = np.max(alpha_index_basic)
    val_pows = np.ones((max_pow + 1, *r_ijs_unit.shape))
    for pow in range(1, max_pow + 1):
        val_pows[pow] = val_pows[pow - 1] * r_ijs_unit
    # Compute basic moments
    for i, aib in enumerate(alpha_index_basic):
        mu, xpow, ypow, zpow = aib
        val = rb_values[mu] * val_pows[xpow, 0] * val_pows[ypow, 1] * val_pows[zpow, 2]
        moment_components[i] = val.sum()
    # Compute contractions
    for ait in alpha_index_times:
        i1, i2, mult, i3 = ait
        moment_components[i3] += mult * moment_components[i1] * moment_components[i2]
    # Compute basis
    basis_vals = moment_components[alpha_moment_mapping]
    return basis_vals
