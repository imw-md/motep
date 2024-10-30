import numpy as np
from ase import Atoms
from ase.neighborlist import PrimitiveNeighborList

from motep.potentials.mtp.data import MTPData


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

    def __init__(self, dict_mtp: MTPData | None = None) -> None:
        """MLIP-2 MTP.

        Parameters
        ----------
        dict_mtp : :class:`motep.potentials.mtp.data.MTPData`
            Parameters in the MLIP .mtp file.

        """
        self.dict_mtp = MTPData()
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
        self.basis_de_dcs = None
        self.basis_ddedcs = None
        self.basis_ds_dcs = None

        # used for `Level2MTPOptimizer`
        self.radial_basis_values = None
        self.radial_basis_dqdris = None
        self.radial_basis_dqdeps = None

    def update(self, dict_mtp: MTPData) -> None:
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
        rfc = self.dict_mtp["radial_funcs_count"]
        rbs = self.dict_mtp["radial_basis_size"]
        asm = self.dict_mtp["alpha_scalar_moments"]

        self.basis_values = np.full((asm), np.nan)
        self.basis_dbdris = np.full((asm, 3, natoms), np.nan)
        self.basis_dbdeps = np.full((asm, 3, 3), np.nan)
        self.basis_de_dcs = np.full((spc, spc, rfc, rbs), np.nan)
        self.basis_ddedcs = np.full((spc, spc, rfc, rbs, 3, natoms), np.nan)
        self.basis_ds_dcs = np.full((spc, spc, rfc, rbs, 3, 3), np.nan)

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
