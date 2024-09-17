"""MTP writtin in Python.

Original version: Axel Forslund
Modified version: Yuji Ikeda
"""

from typing import Any

import numpy as np
from ase import Atoms
from ase.neighborlist import PrimitiveNeighborList
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


class NumpyMTPEngine:
    def __init__(self, mtp_parameters: dict[str, Any] | None = None):
        """MLIP-2 MTP.

        Parameters
        ----------
        mtp_parameters : dict[str, Any]
            Parameters in the MLIP .mtp file.

        """
        self.parameters = {}
        if mtp_parameters is not None:
            self.update(mtp_parameters)
        self.results = {}
        self._neighbor_list = None

    def update(self, parameters: dict[str, Any]) -> None:
        self.parameters = parameters
        if "species" not in self.parameters:
            species = {_: _ for _ in range(self.parameters["species_count"])}
            self.parameters["species"] = species
        if "radial_coeffs" in self.parameters:
            self.radial_basis_funcs = init_radial_basis_functions(
                self.parameters["radial_coeffs"],
                self.parameters["min_dist"],
                self.parameters["max_dist"],
            )

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

    def _get_local_energy(self, atoms: Atoms, i: int, js: list[int], r_ijs):
        itype = self.parameters["species"][atoms.numbers[i]]
        jtypes = [self.parameters["species"][atoms.numbers[j]] for j in js]
        r_abs = np.linalg.norm(r_ijs, axis=0)
        rb_values, rb_derivs = self.calc_radial_basis(r_abs, itype, jtypes)
        basis_values, basis_derivs = calc_moment_basis(
            r_ijs,
            r_abs,
            rb_values,
            rb_derivs,
            self.parameters["alpha_moments_count"],
            self.parameters["alpha_index_basic"],
            self.parameters["alpha_index_times"],
            self.parameters["alpha_moment_mapping"],
        )
        moment_coeffs = self.parameters["moment_coeffs"]
        return (
            moment_coeffs @ basis_values,
            np.tensordot(moment_coeffs, basis_derivs, axes=(0, 0)),
        )

    def get_energy(self, atoms: Atoms):
        """Calculate the energy of the given system."""
        self.update_neighbor_list(atoms)
        energies = np.zeros(len(atoms))
        forces = np.zeros((len(atoms), 3))
        stress = np.zeros((3, 3))
        for i in range(len(atoms)):
            js, r_ijs = self._get_distances(atoms, i)
            e, gradient = self._get_local_energy(atoms, i, js, r_ijs)
            itype = self.parameters["species"][atoms.numbers[i]]
            energies[i] = e + self.parameters["species_coeffs"][itype]
            for k, j in enumerate(js):
                forces[i] -= gradient[:, k]
                forces[j] += gradient[:, k]
            stress += r_ijs @ gradient.T
        self.results["energies"] = energies
        self.results["energy"] = self.results["energies"].sum()
        self.results["forces"] = forces

        if atoms.cell.rank == 3:
            stress = (stress + stress.T) * 0.5  # symmetrize
            stress /= atoms.get_volume()
            self.results["stress"] = stress.flat[[0, 4, 8, 5, 2, 1]]
        else:
            self.results["stress"] = np.full(6, np.nan)

        return self.results["energy"], self.results["forces"], self.results["stress"]

    def _initiate_neighbor_list(self, atoms: Atoms):
        self._neighbor_list = PrimitiveNeighborList(
            cutoffs=[self.parameters["max_dist"]] * len(atoms),
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


def _compute_offsets(nl: PrimitiveNeighborList, atoms: Atoms):
    cell = atoms.cell
    return [nl.get_neighbors(j)[1] @ cell for j in range(len(atoms))]


def calc_moment_basis(
    r_ijs: np.ndarray,  # (3, neighbors)
    r_abs: np.ndarray,  # (neighbors)
    rb_values: np.ndarray,  # (mu, neighbors)
    rb_derivs: np.ndarray,  # (mu, neighbors)
    alpha_moments_count: int,
    alpha_index_basic: int,
    alpha_index_times: int,
    alpha_moment_mapping: np.ndarray,
):
    moment_components = np.zeros(alpha_moments_count)
    moment_jacobian = np.zeros((alpha_moments_count, *r_ijs.shape))  # dEi/dxj
    # Precompute powers
    max_pow = np.max(alpha_index_basic)
    abs_pows = np.ones((max_pow + 1, *r_abs.shape))
    val_pows = np.ones((max_pow + 1, *r_ijs.shape))
    for pow in range(1, max_pow + 1):
        abs_pows[pow] = abs_pows[pow - 1] * r_abs
        val_pows[pow] = val_pows[pow - 1] * r_ijs
    # Compute basic moments
    for i, aib in enumerate(alpha_index_basic):
        mu, xpow, ypow, zpow = aib
        k = xpow + ypow + zpow
        mult0 = (
            1.0
            * val_pows[xpow, 0]
            * val_pows[ypow, 1]
            * val_pows[zpow, 2]
            / abs_pows[k]
        )
        val = rb_values[mu] * mult0
        der = rb_derivs[mu] * mult0 * r_ijs / r_abs
        der -= val * k * r_ijs / (r_abs**2)
        if xpow != 0:
            der[0] += (
                rb_values[mu]
                * (xpow * val_pows[xpow - 1, 0])
                * val_pows[ypow, 1]
                * val_pows[zpow, 2]
            ) / abs_pows[k]
        if ypow != 0:
            der[1] += (
                rb_values[mu]
                * val_pows[xpow, 0]
                * (ypow * val_pows[ypow - 1, 1])
                * val_pows[zpow, 2]
            ) / abs_pows[k]
        if zpow != 0:
            der[2] += (
                rb_values[mu]
                * val_pows[xpow, 0]
                * val_pows[ypow, 1]
                * (zpow * val_pows[zpow - 1, 2])
            ) / abs_pows[k]
        moment_components[i] = val.sum()
        moment_jacobian[i] = der
    # Compute contractions
    for ait in alpha_index_times:
        i1, i2, mult, i3 = ait
        moment_components[i3] += mult * moment_components[i1] * moment_components[i2]
        moment_jacobian[i3] += mult * moment_jacobian[i1] * moment_components[i2]
        moment_jacobian[i3] += mult * moment_components[i1] * moment_jacobian[i2]
    # Compute basis
    basis_vals = moment_components[alpha_moment_mapping]
    basis_ders = moment_jacobian[alpha_moment_mapping]
    return basis_vals, basis_ders
