"""MTP writtin in Python.

Original version: Axel Forslund
Modified version: Yuji Ikeda
"""

from typing import Any

import numpy as np
import numpy.typing as npt
from ase import Atoms
from ase.neighborlist import PrimitiveNeighborList
from numpy.polynomial import Chebyshev


class EngineBase:
    """Engine to compute an MTP.

    Attributes
    ----------
    basis_values : np.ndarray (alpha_moments_count)
        Basis values summed over atoms.
        This corresponds to b_j in Eq. (5) in [Podryabinkin_CMS_2017_Active]_.
    basis_derivs : np.ndarray (alpha_moments_count, 3, number_of_atoms)
        Derivatives of basis functions with respect to Cartesian coordinates of atoms
        summed over atoms.
        This corresponds to nabla b_j in Eq. (7a) in [Podryabinkin_CMS_2017_Active]_.

    .. [Podryabinkin_CMS_2017_Active]
       E. V. Podryabinkin and A. V. Shapeev, Comput. Mater. Sci. 140, 171 (2017).

    """

    def __init__(self, dict_mtp: dict[str, Any] | None = None) -> None:
        """MLIP-2 MTP.

        Parameters
        ----------
        dict_mtp : dict[str, Any]
            Parameters in the MLIP .mtp file.

        """
        self.dict_mtp = {}
        if dict_mtp is not None:
            self.update(dict_mtp)
        self.results = {}
        self._neighbor_list = None
        self.basis_values = None
        self.basis_derivs = None

    def update(self, dict_mtp: dict[str, Any]) -> None:
        """Update MTP parameters."""
        self.dict_mtp = dict_mtp
        if "species" not in self.dict_mtp:
            species = {_: _ for _ in range(self.dict_mtp["species_count"])}
            self.dict_mtp["species"] = species

    def update_neighbor_list(self, atoms: Atoms) -> None:
        """Update the ASE `PrimitiveNeighborList` object."""
        if self._neighbor_list is None:
            self._initiate_neighbor_list(atoms)
        elif self._neighbor_list.update(atoms.pbc, atoms.cell, atoms.positions):
            self.precomputed_offsets = _compute_offsets(self._neighbor_list, atoms)

    def _initiate_neighbor_list(self, atoms: Atoms) -> None:
        """Initialize the ASE `PrimitiveNeighborList` object."""
        self._neighbor_list = PrimitiveNeighborList(
            cutoffs=[0.5 * self.dict_mtp["max_dist"]] * len(atoms),
            skin=0.3,  # cutoff + skin is used, recalc only if diff in pos > skin
            self_interaction=False,  # Exclude [0, 0, 0]
            bothways=True,  # return both ij and ji
        )
        self._neighbor_list.update(atoms.pbc, atoms.cell, atoms.positions)
        self.precomputed_offsets = _compute_offsets(self._neighbor_list, atoms)

        self.energies = np.full(len(atoms), np.nan)
        self.forces = np.full((len(atoms), 3), np.nan)
        self.stress = np.full((3, 3), np.nan)

        shape = self.dict_mtp["alpha_scalar_moments"]
        self.basis_values = np.full(shape, np.nan)

        number_of_atoms = len(atoms)
        shape = self.dict_mtp["alpha_scalar_moments"], 3, number_of_atoms
        self.basis_derivs = np.full(shape, np.nan)

    def _get_distances(
        self,
        atoms: Atoms,
        index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        indices_js, _ = self._neighbor_list.get_neighbors(index)
        offsets = self.precomputed_offsets[index]
        pos_js = atoms.positions[indices_js] + offsets
        dist_vectors = pos_js - atoms.positions[index]
        return indices_js, dist_vectors.T


def _compute_offsets(nl: PrimitiveNeighborList, atoms: Atoms):
    cell = atoms.cell
    return [nl.get_neighbors(j)[1] @ cell for j in range(len(atoms))]


class NumpyMTPEngine(EngineBase):
    def __init__(self, dict_mtp: dict[str, Any] | None = None):
        """MLIP-2 MTP.

        Parameters
        ----------
        dict_mtp : dict[str, Any]
            Parameters in the MLIP .mtp file.

        """
        self.radial_basis_funcs = None
        self.radial_basis_dfdrs = None
        super().__init__(dict_mtp)

    def update(self, dict_mtp: dict[str, Any]) -> None:
        """Update MTP parameters."""
        super().update(dict_mtp)
        if "radial_coeffs" in self.dict_mtp:
            if self.radial_basis_funcs is None:
                self.radial_basis_funcs, self.radial_basis_dfdrs = (
                    init_radial_basis_functions(
                        self.dict_mtp["radial_coeffs"],
                        self.dict_mtp["min_dist"],
                        self.dict_mtp["max_dist"],
                    )
                )
            else:
                update_radial_basis_coefficients(
                    self.dict_mtp["radial_coeffs"],
                    self.radial_basis_funcs,
                    self.radial_basis_dfdrs,
                )

    def _calc_radial_basis(
        self,
        r_abs: np.ndarray,
        itype: int,
        jtypes: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        return calc_radial_basis(
            self.radial_basis_funcs,
            self.radial_basis_dfdrs,
            r_abs,
            itype,
            jtypes,
            self.dict_mtp["scaling"],
            self.dict_mtp["max_dist"],
            self.dict_mtp["radial_funcs_count"],
        )

    def _calc_basis(
        self,
        atoms: Atoms,
        i: int,
        js: list[int],
        r_ijs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        itype = self.dict_mtp["species"][atoms.numbers[i]]
        jtypes = [self.dict_mtp["species"][atoms.numbers[j]] for j in js]
        r_abs = np.sqrt(np.add.reduce(r_ijs**2, axis=0))
        rb_values, rb_derivs = self._calc_radial_basis(r_abs, itype, jtypes)
        return calc_moment_basis(
            r_ijs,
            r_abs,
            rb_values,
            rb_derivs,
            self.dict_mtp["alpha_moments_count"],
            self.dict_mtp["alpha_index_basic"],
            self.dict_mtp["alpha_index_times"],
            self.dict_mtp["alpha_moment_mapping"],
        )

    def calculate(self, atoms: Atoms) -> tuple:
        """Calculate properties of the given system."""
        self.update_neighbor_list(atoms)
        self.energies[:] = 0.0
        self.forces[:, :] = 0.0
        self.stress[:, :] = 0.0
        self.basis_values[:] = 0.0
        self.basis_derivs[:, :, :] = 0.0

        moment_coeffs = self.dict_mtp["moment_coeffs"]

        for i in range(len(atoms)):
            js, r_ijs = self._get_distances(atoms, i)
            basis_values, basis_derivs = self._calc_basis(atoms, i, js, r_ijs)

            self.basis_values += basis_values

            site_energy = moment_coeffs @ basis_values
            gradient = np.tensordot(moment_coeffs, basis_derivs, axes=(0, 0))

            itype = self.dict_mtp["species"][atoms.numbers[i]]
            self.energies[i] = site_energy + self.dict_mtp["species_coeffs"][itype]
            # Calculate forces
            # Be careful that the derivative of the site energy of the j-th atom also
            # contributes to the forces on the i-th atom.
            # Be also careful that:
            # 1. In `calc_moment_basis`, the derivatives with respect to the j-th atom
            #    (not the center i-th) atom is computed.
            # 2. The force on the i-th atom is defined as the negative of the gradient
            #    with respect to the i-th atom.
            # Thus, the negative signs of the two contributions are cancelled out below.
            for k, j in enumerate(js):
                self.forces[i] += gradient[:, k]
                self.forces[j] -= gradient[:, k]
                self.basis_derivs[:, :, i] -= basis_derivs[:, :, k]
                self.basis_derivs[:, :, j] += basis_derivs[:, :, k]
            self.stress += r_ijs @ gradient.T
        self.results["energies"] = self.energies
        self.results["energy"] = self.results["energies"].sum()
        self.results["forces"] = self.forces

        if atoms.cell.rank == 3:
            self.stress = (self.stress + self.stress.T) * 0.5  # symmetrize
            self.stress /= atoms.get_volume()
        else:
            self.stress[:, :] = np.nan
        self.results["stress"] = self.stress.flat[[0, 4, 8, 5, 2, 1]]

        return self.results["energy"], self.results["forces"], self.results["stress"]


#
# Numpy implemented functions for radial basis and moment basis evaluation:
#
def init_radial_basis_functions(
    radial_coeffs: np.ndarray,
    min_dist: float,
    max_dist: float,
) -> tuple[np.ndarray, np.ndarray]:  # array of Chebyshev objects
    """Initialize radial basis functions."""
    radial_basis_funcs = []
    radial_basis_dfdrs = []  # derivatives
    domain = [min_dist, max_dist]
    nspecies, _, nmu, _ = radial_coeffs.shape
    for i0 in range(nspecies):
        for i1 in range(nspecies):
            for i2 in range(nmu):
                p = Chebyshev(radial_coeffs[i0, i1, i2], domain=domain)
                radial_basis_funcs.append(p)
                radial_basis_dfdrs.append(p.deriv())
    shape = nspecies, nspecies, nmu
    return (
        np.array(radial_basis_funcs).reshape(shape),
        np.array(radial_basis_dfdrs).reshape(shape),
    )


def update_radial_basis_coefficients(
    radial_coeffs: np.ndarray,
    radial_basis_funcs: np.ndarray,  # array of Chebyshev objects
    radial_basis_dfdrs: np.ndarray,  # array of Chebyshev objects
) -> None:
    """Update radial basis coefficients."""
    nspecies, _, nmu, _ = radial_coeffs.shape
    for i0 in range(nspecies):
        for i1 in range(nspecies):
            for i2 in range(nmu):
                p = radial_basis_funcs[i0, i1, i2]
                p.coef = radial_coeffs[i0, i1, i2]
                radial_basis_dfdrs[i0, i1, i2] = p.deriv()


def calc_radial_basis(
    radial_basis_funcs: np.ndarray,  # array of Chebyshev objects
    radial_basis_dfdrs: np.ndarray,  # array of Chebyshev objects
    r_abs: np.ndarray,
    itype: int,
    jtypes: list[int],
    scaling: float,
    max_dist: float,
    radial_funcs_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    is_within_cutoff = r_abs < max_dist
    smooth_values = scaling * (max_dist - r_abs) ** 2
    smooth_derivs = -2.0 * scaling * (max_dist - r_abs)
    rb_values = np.zeros((radial_funcs_count, r_abs.size))
    rb_derivs = np.zeros((radial_funcs_count, r_abs.size))
    for mu in range(radial_funcs_count):
        for j, jtype in enumerate(jtypes):
            if is_within_cutoff[j]:
                rb_func = radial_basis_funcs[itype, jtype, mu]
                rb_dfdr = radial_basis_dfdrs[itype, jtype, mu]
                v = rb_func(r_abs[j]) * smooth_values[j]
                rb_values[mu, j] = v
                d0 = rb_dfdr(r_abs[j]) * smooth_values[j]
                d1 = rb_func(r_abs[j]) * smooth_derivs[j]
                d = d0 + d1
                rb_derivs[mu, j] = d
    return rb_values, rb_derivs


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


#
# Class for Numba implementation
#
class NumbaMTPEngine(EngineBase):
    """MTP Engine based on Numba."""

    def calculate(self, atoms: Atoms):
        self.update_neighbor_list(atoms)
        return self.numba_calc_energy_and_forces(atoms)

    def numba_calc_energy_and_forces(self, atoms):
        from motep.numba import numba_calc_energy_and_forces

        mlip_params = self.dict_mtp
        energy, forces, stress = numba_calc_energy_and_forces(
            self,
            atoms,
            mlip_params["alpha_moments_count"],
            mlip_params["alpha_moment_mapping"],
            mlip_params["alpha_index_basic"],
            mlip_params["alpha_index_times"],
            mlip_params["scaling"],
            mlip_params["min_dist"],
            mlip_params["max_dist"],
            mlip_params["species_coeffs"],
            mlip_params["moment_coeffs"],
            mlip_params["radial_coeffs"],
        )
        return energy, forces, stress
