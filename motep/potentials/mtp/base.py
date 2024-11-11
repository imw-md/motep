from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from ase import Atoms
from ase.neighborlist import PrimitiveNeighborList

from motep.potentials.mtp.data import MTPData


@dataclass
class MomentBasisData:
    """Data related to the moment basis.

    Attributes
    ----------
    values : np.ndarray (alpha_moments_count)
        Basis values summed over atoms.
        This corresponds to b_j in Eq. (5) in [Podryabinkin_CMS_2017_Active]_.
    dbdris : np.ndarray (alpha_moments_count, 3, number_of_atoms)
        Derivatives of basis functions with respect to Cartesian coordinates of atoms
        summed over atoms.
        This corresponds to nabla b_j in Eq. (7a) in [Podryabinkin_CMS_2017_Active]_.
    dbdeps : np.ndarray (alpha_moments_count, 3, 3)
        Derivatives of cumulated basis functions with respect to the strain tensor.

    .. [Podryabinkin_CMS_2017_Active]
       E. V. Podryabinkin and A. V. Shapeev, Comput. Mater. Sci. 140, 171 (2017).

    """

    values: npt.NDArray[np.float64] | None = None
    dbdris: npt.NDArray[np.float64] | None = None
    dbdeps: npt.NDArray[np.float64] | None = None
    dedcs: npt.NDArray[np.float64] | None = None
    dgdcs: npt.NDArray[np.float64] | None = None
    dsdcs: npt.NDArray[np.float64] | None = None

    def initialize(self, natoms: int, mtp_data: MTPData) -> None:
        """Initialize moment basis properties."""
        spc = mtp_data["species_count"]
        rfc = mtp_data["radial_funcs_count"]
        rbs = mtp_data["radial_basis_size"]
        asm = mtp_data["alpha_scalar_moments"]

        self.values = np.full((asm), np.nan)
        self.dbdris = np.full((asm, natoms, 3), np.nan)
        self.dbdeps = np.full((asm, 3, 3), np.nan)
        self.dedcs = np.full((spc, spc, rfc, rbs), np.nan)
        self.dgdcs = np.full((spc, spc, rfc, rbs, natoms, 3), np.nan)
        self.dsdcs = np.full((spc, spc, rfc, rbs, 3, 3), np.nan)

    def clean(self) -> None:
        """Clean up moment basis properties."""
        self.values[...] = 0.0
        self.dbdris[...] = 0.0
        self.dbdeps[...] = 0.0
        self.dedcs[...] = 0.0
        self.dgdcs[...] = 0.0
        self.dsdcs[...] = 0.0


@dataclass
class RadialBasisData:
    """Data related to the radial basis.

    Attributes
    ----------
    values : np.ndarray (species_count, species_count, radial_basis_size)
        Radial basis values summed over atoms.
    dqdris : (species_count, species_count, radial_basis_size, 3, natoms)
        Derivaties of radial basis functions summed over atoms.

    """

    values: npt.NDArray[np.float64] | None = None
    dqdris: npt.NDArray[np.float64] | None = None
    dqdeps: npt.NDArray[np.float64] | None = None

    def initialize(self, natoms: int, mtp_data: MTPData) -> None:
        """Initialize radial basis properties."""
        spc = mtp_data["species_count"]
        rbs = mtp_data["radial_basis_size"]

        self.values = np.full((spc, spc, rbs), np.nan)
        self.dqdris = np.full((spc, spc, rbs, natoms, 3), np.nan)
        self.dqdeps = np.full((spc, spc, rbs, 3, 3), np.nan)

    def clean(self) -> None:
        """Clean up radial basis properties."""
        self.values[...] = 0.0
        self.dqdris[...] = 0.0
        self.dqdeps[...] = 0.0


class Jac(dict):
    @property
    def parameters(self) -> npt.NDArray[np.float64]:
        shape = self["radial_coeffs"].shape
        return np.concatenate(
            (
                self["scaling"],
                self["moment_coeffs"],
                self["species_coeffs"],
                self["radial_coeffs"].reshape(-1, *shape[4::]),
            ),
        )


class EngineBase:
    """Engine to compute an MTP."""

    def __init__(
        self,
        mtp_data: MTPData,
        *,
        is_trained: bool = False,
    ) -> None:
        """MLIP-2 MTP.

        Parameters
        ----------
        mtp_data : :class:`motep.potentials.mtp.data.MTPData`
            Parameters in the MLIP .mtp file.
        is_trained : bool, default False
            If True, basis data for training is computed and stored.

        """
        self.update(mtp_data)
        self.results = {}
        self._neighbor_list = None
        self._is_trained = is_trained

        # moment basis data
        self.mbd = MomentBasisData()

        # used for `Level2MTPOptimizer`
        self.rbd = RadialBasisData()

    def update(self, mtp_data: MTPData) -> None:
        """Update MTP parameters."""
        self.mtp_data = mtp_data
        if "species" not in self.mtp_data:
            self.mtp_data["species"] = list(range(self.mtp_data["species_count"]))

    def update_neighbor_list(self, atoms: Atoms) -> None:
        """Update the ASE `PrimitiveNeighborList` object."""
        if self._neighbor_list is None:
            self._initiate_neighbor_list(atoms)
        elif self._neighbor_list.update(atoms.pbc, atoms.cell, atoms.positions):
            self.precomputed_offsets = _compute_offsets(self._neighbor_list, atoms)

    def _initiate_neighbor_list(self, atoms: Atoms) -> None:
        """Initialize the ASE `PrimitiveNeighborList` object."""
        self._neighbor_list = PrimitiveNeighborList(
            cutoffs=[0.5 * self.mtp_data["max_dist"]] * len(atoms),
            skin=0.3,  # cutoff + skin is used, recalc only if diff in pos > skin
            self_interaction=False,  # Exclude [0, 0, 0]
            bothways=True,  # return both ij and ji
        )
        self._neighbor_list.update(atoms.pbc, atoms.cell, atoms.positions)
        self.precomputed_offsets = _compute_offsets(self._neighbor_list, atoms)

        natoms = len(atoms)

        self.mbd.initialize(natoms, self.mtp_data)
        self.rbd.initialize(natoms, self.mtp_data)

    def _get_distances(
        self,
        atoms: Atoms,
        index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        indices_js, _ = self._neighbor_list.get_neighbors(index)
        offsets = self.precomputed_offsets[index]
        pos_js = atoms.positions[indices_js] + offsets
        dist_vectors = pos_js - atoms.positions[index]
        return indices_js, dist_vectors

    def _symmetrize_stress(self, atoms: Atoms, stress: np.ndarray) -> None:
        if atoms.cell.rank == 3:
            volume = atoms.get_volume()
            stress += stress.T
            stress *= 0.5 / volume
            self.mbd.dbdeps += self.mbd.dbdeps.transpose(0, 2, 1)
            self.mbd.dbdeps *= 0.5 / volume
            self.mbd.dsdcs += self.mbd.dsdcs.swapaxes(-2, -1)
            self.mbd.dsdcs *= 0.5 / volume
            axes = 0, 1, 2, 4, 3
            self.rbd.dqdeps += self.rbd.dqdeps.transpose(axes)
            self.rbd.dqdeps *= 0.5 / volume
        else:
            stress[:, :] = np.nan
            self.mbd.dbdeps[:, :, :] = np.nan
            self.rbd.dqdeps[:, :, :] = np.nan

    def jac_energy(self, atoms: Atoms) -> MTPData:
        """Calculate the Jacobian of the energy with respect to the MTP parameters."""
        sps = self.mtp_data["species"]
        nbs = list(atoms.numbers)

        jac = MTPData()  # placeholder of the Jacobian with respect to the parameters
        jac["scaling"] = 0.0
        jac["moment_coeffs"] = self.mbd.values.copy()
        jac["species_coeffs"] = np.fromiter((nbs.count(s) for s in sps), dtype=float)
        jac["radial_coeffs"] = self.mbd.dedcs.copy()

        return jac

    def jac_forces(self, atoms: Atoms) -> MTPData:
        """Calculate the Jacobian of the forces with respect to the MTP parameters.

        `jac.parameters` have the shape of `(nparams, natoms, 3)`.

        """
        spc = self.mtp_data["species_count"]
        number_of_atoms = len(atoms)

        jac = Jac()  # placeholder of the Jacobian with respect to the parameters
        jac["scaling"] = np.zeros((1, number_of_atoms, 3))
        jac["moment_coeffs"] = self.mbd.dbdris * -1.0
        jac["species_coeffs"] = np.zeros((spc, number_of_atoms, 3))
        jac["radial_coeffs"] = self.mbd.dgdcs * -1.0

        return jac

    def jac_stress(self, atoms: Atoms) -> MTPData:
        """Calculate the Jacobian of the forces with respect to the MTP parameters.

        `jac.parameters` have the shape of `(nparams, natoms, 3)`.

        """
        spc = self.mtp_data["species_count"]

        jac = Jac()  # placeholder of the Jacobian with respect to the parameters
        jac["scaling"] = np.zeros((1, 3, 3))
        jac["moment_coeffs"] = self.mbd.dbdeps.copy()
        jac["species_coeffs"] = np.zeros((spc, 3, 3))
        jac["radial_coeffs"] = self.mbd.dsdcs.copy()

        return jac


def _compute_offsets(nl: PrimitiveNeighborList, atoms: Atoms):
    cell = atoms.cell
    return [nl.get_neighbors(j)[1] @ cell for j in range(len(atoms))]
