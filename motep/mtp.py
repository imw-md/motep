"""MTP writtin in Python.

Original version: Axel Forslund
Modified version: Yuji Ikeda
"""

from typing import Any

import numpy as np
import numpy.typing as npt
from ase import Atoms
from ase.neighborlist import PrimitiveNeighborList


class EngineBase:
    """Engine to compute an MTP.

    Attributes
    ----------
    basis_values : np.ndarray (alpha_moments_count)
        Basis values summed over atoms.
        This corresponds to b_j in Eq. (5) in [Podryabinkin_CMS_2017_Active]_.
    basis_dbdris : np.ndarray (alpha_moments_count, 3, number_of_atoms)
        Derivatives of basis functions with respect to Cartesian coordinates of atoms
        summed over atoms.
        This corresponds to nabla b_j in Eq. (7a) in [Podryabinkin_CMS_2017_Active]_.
    basis_dbdeps : np.ndarray (alpha_moments_count, 3, 3)
        Derivatives of cumulated basis functions with respect to the strain tensor.
    radial_basis_values : np.ndarray (species_count, species_count, radial_basis_size)
        Radial basis values summed over atoms.
    radial_basis_derivs : (species_count, species_count, radial_basis_size, 3, natoms)
        Derivaties of radial basis functions summed over atoms.

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

        self.energies = None
        self.forces = None
        self.stress = None

        self.basis_values = None
        self.basis_dbdris = None
        self.basis_dbdeps = None

        # used for `Level2MTPOptimizer`
        self.radial_basis_values = None
        self.radial_basis_dqdris = None
        self.radial_basis_dqdeps = None

    def update(self, dict_mtp: dict[str, Any]) -> None:
        """Update MTP parameters."""
        self.dict_mtp = dict_mtp
        if "species" not in self.dict_mtp:
            self.dict_mtp["species"] = list(range(self.dict_mtp["species_count"]))

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

        natoms = len(atoms)
        spc = self.dict_mtp["species_count"]
        rbs = self.dict_mtp["radial_basis_size"]
        asm = self.dict_mtp["alpha_scalar_moments"]

        self.basis_values = np.full((asm), np.nan)
        self.basis_dbdris = np.full((asm, 3, natoms), np.nan)
        self.basis_dbdeps = np.full((asm, 3, 3), np.nan)

        self.radial_basis_values = np.full((spc, spc, rbs), np.nan)
        self.radial_basis_dqdris = np.full((spc, spc, rbs, 3, natoms), np.nan)
        self.radial_basis_dqdeps = np.full((spc, spc, rbs, 3, 3), np.nan)

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
    """MTP engine based on NumPy."""

    def __init__(self, dict_mtp: dict[str, Any] | None = None) -> None:
        """Intialize the engine.

        Parameters
        ----------
        dict_mtp : dict[str, Any]
            Parameters in the MLIP .mtp file.

        """
        from motep.radial import ChebyshevArrayRadialBasis

        self.rb = ChebyshevArrayRadialBasis(dict_mtp)
        super().__init__(dict_mtp)

    def update(self, dict_mtp: dict[str, Any]) -> None:
        """Update MTP parameters."""
        super().update(dict_mtp)
        if "radial_coeffs" in self.dict_mtp:
            self.rb.update_coeffs(self.dict_mtp["radial_coeffs"])

    def _calc_basis(
        self,
        atoms: Atoms,
        i: int,
        js: list[int],
        r_ijs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        itype = self.dict_mtp["species"].index(atoms.numbers[i])
        jtypes = [self.dict_mtp["species"].index(atoms.numbers[j]) for j in js]
        r_abs = np.sqrt(np.add.reduce(r_ijs**2, axis=0))
        r_ijs_unit = r_ijs / r_abs

        rb_values, rb_derivs = self.rb.calculate(r_abs, itype, jtypes)
        np.add.at(self.radial_basis_values[itype], jtypes, self.rb.values0[:, :])
        for k, (j, jtype) in enumerate(zip(js, jtypes, strict=True)):
            tmp = self.rb.derivs0[k, :, None] * r_ijs_unit[:, k]
            self.radial_basis_dqdris[itype, jtype, :, :, i] -= tmp
            self.radial_basis_dqdris[itype, jtype, :, :, j] += tmp
            self.radial_basis_dqdeps[itype, jtype] += tmp[:, :, None] * r_ijs[:, k]

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
        itypes = [self.dict_mtp["species"].index(_) for _ in atoms.numbers]
        self.energies = self.dict_mtp["species_coeffs"][itypes]

        self.basis_values[:] = 0.0
        self.basis_dbdris[:, :, :] = 0.0
        self.basis_dbdeps[:, :, :] = 0.0

        self.radial_basis_values[...] = 0.0
        self.radial_basis_dqdris[...] = 0.0
        self.radial_basis_dqdeps[...] = 0.0

        moment_coeffs = self.dict_mtp["moment_coeffs"]

        for i in range(len(atoms)):
            js, r_ijs = self._get_distances(atoms, i)
            basis_values, basis_derivs = self._calc_basis(atoms, i, js, r_ijs)

            self.basis_values += basis_values

            self.energies[i] += moment_coeffs @ basis_values
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
                self.basis_dbdris[:, :, i] -= basis_derivs[:, :, k]
                self.basis_dbdris[:, :, j] += basis_derivs[:, :, k]
            self.basis_dbdeps += basis_derivs @ r_ijs.T

        self.forces = np.sum(moment_coeffs * self.basis_dbdris.T, axis=-1) * -1.0
        self.stress = np.sum(moment_coeffs * self.basis_dbdeps.T, axis=-1).T

        self.results["energies"] = self.energies
        self.results["energy"] = self.results["energies"].sum()
        self.results["forces"] = self.forces

        if atoms.cell.rank == 3:
            self.stress = (self.stress + self.stress.T) * 0.5  # symmetrize
            self.stress /= atoms.get_volume()
            self.basis_dbdeps += self.basis_dbdeps.transpose(0, 2, 1)
            self.basis_dbdeps *= 0.5 / atoms.get_volume()
            axes = 0, 1, 2, 4, 3
            self.radial_basis_dqdeps += self.radial_basis_dqdeps.transpose(axes)
            self.radial_basis_dqdeps *= 0.5 / atoms.get_volume()
        else:
            self.stress[:, :] = np.nan
            self.basis_dbdeps[:, :, :] = np.nan
            self.radial_basis_dqdeps[:, :, :] = np.nan

        self.results["stress"] = self.stress.flat[[0, 4, 8, 5, 2, 1]]

        return self.results["energy"], self.results["forces"], self.results["stress"]


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
