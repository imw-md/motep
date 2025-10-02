from abc import abstractmethod
from dataclasses import dataclass, field

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
        spc = mtp_data.species_count
        rfc = mtp_data.radial_funcs_count
        rbs = mtp_data.radial_basis_size
        asm = mtp_data.alpha_scalar_moments

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
        spc = mtp_data.species_count
        rbs = mtp_data.radial_basis_size

        self.values = np.full((spc, spc, rbs), np.nan)
        self.dqdris = np.full((spc, spc, rbs, natoms, 3), np.nan)
        self.dqdeps = np.full((spc, spc, rbs, 3, 3), np.nan)

    def clean(self) -> None:
        """Clean up radial basis properties."""
        self.values[...] = 0.0
        self.dqdris[...] = 0.0
        self.dqdeps[...] = 0.0


@dataclass
class Jac:
    scaling: float = 1.0
    radial_coeffs: npt.NDArray[np.float64] | None = None
    species_coeffs: npt.NDArray[np.float64] | None = None
    moment_coeffs: npt.NDArray[np.float64] | None = None
    optimized: list[str] = field(
        default_factory=lambda: ["species_coeffs", "moment_coeffs", "radial_coeffs"],
    )

    @property
    def parameters(self) -> npt.NDArray[np.float64]:
        """Serialized parameters."""
        tmp = []
        if "scaling" in self.optimized:
            tmp.append(np.atleast_1d(self.scaling))
        if "moment_coeffs" in self.optimized:
            tmp.append(self.moment_coeffs)
        if "species_coeffs" in self.optimized:
            tmp.append(self.species_coeffs)
        if "radial_coeffs" in self.optimized:
            shape = self.radial_coeffs.shape
            tmp.append(self.radial_coeffs.reshape(-1, *shape[4::]))
        return np.concatenate(tmp)


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
        if self.mtp_data.species is None:
            self.mtp_data.species = list(range(self.mtp_data.species_count))

    def update_neighbor_list(self, atoms: Atoms) -> None:
        """Update the ASE `PrimitiveNeighborList` object."""
        if self._neighbor_list is None:
            self._initiate_neighbor_list(atoms)
        elif self._neighbor_list.update(atoms.pbc, atoms.cell, atoms.positions):
            self.precomputed_offsets = _compute_offsets(self._neighbor_list, atoms)
            all_precomp = _compute_all_offsets(self._neighbor_list, atoms)
            self.all_js, self.all_offsets = all_precomp

    def _initiate_neighbor_list(self, atoms: Atoms) -> None:
        """Initialize the ASE `PrimitiveNeighborList` object."""
        self._neighbor_list = PrimitiveNeighborList(
            cutoffs=[0.5 * self.mtp_data.max_dist] * len(atoms),
            skin=0.3,  # cutoff + skin is used, recalc only if diff in pos > skin
            self_interaction=False,  # Exclude [0, 0, 0]
            bothways=True,  # return both ij and ji
        )
        self._neighbor_list.update(atoms.pbc, atoms.cell, atoms.positions)
        self.precomputed_offsets = _compute_offsets(self._neighbor_list, atoms)
        all_precomp = _compute_all_offsets(self._neighbor_list, atoms)
        self.all_js, self.all_offsets = all_precomp

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

    def _get_all_distances(self, atoms: Atoms) -> tuple[np.ndarray, np.ndarray]:
        max_dist = self.mtp_data.max_dist
        positions = atoms.positions
        offsets = self.all_offsets
        all_r_ijs = positions[self.all_js] + offsets - positions[:, None, :]
        all_r_ijs[self.all_js[:, :] < 0, :] = max_dist
        return self.all_js, all_r_ijs

    def check_species(self, atoms: Atoms) -> None:
        """Check if `atoms` comply with the `mtp_data.species`.

        Raises
        ------
        ValueError
            If the unique `atoms.numbers` is larger than `species_count`, or if
            they are not present in `mtp_data.species` (only when not
            `_is_trained`).

        """
        if np.unique(atoms.numbers).size > self.mtp_data.species_count:
            msg = "The number of species in input atoms is larger than species_count."
            raise ValueError(msg)
        if not self._is_trained:
            unique_species = np.unique(atoms.numbers)
            if not all(_ in self.mtp_data.species for _ in unique_species):
                msg = (
                    "All species in input atoms are not in mtp_data.species.\n"
                    f"  species in atoms: {unique_species}\n"
                    f"  species in mtp_data: {self.mtp_data.species}"
                )
                raise ValueError(msg)

    @abstractmethod
    def _calculate(self, atoms: Atoms) -> tuple: ...

    def calculate(self, atoms: Atoms) -> dict:
        """Calculate properties of the given system.

        Returns
        -------
        Dictionary with energies, energy, forces and stress.

        """
        self.check_species(atoms)
        self.update_neighbor_list(atoms)

        energies, forces, stress = self._calculate(atoms)

        self._symmetrize_stress(atoms, stress)

        self.results["energies"] = energies
        self.results["energy"] = self.results["energies"].sum()
        self.results["forces"] = forces
        self.results["stress"] = stress.flat[[0, 4, 8, 5, 2, 1]]

        return self.results

    def _symmetrize_stress(self, atoms: Atoms, stress: np.ndarray) -> None:
        if atoms.cell.rank == 3:  # noqa: PLR2004
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
        sps = self.mtp_data.species
        nbs = list(atoms.numbers)

        jac = MTPData(
            scaling=0.0,
            moment_coeffs=self.mbd.values.copy(),
            species_coeffs=np.fromiter((nbs.count(s) for s in sps), dtype=float),
            radial_coeffs=self.mbd.dedcs.copy(),
        )  # placeholder of the Jacobian with respect to the parameters
        jac.optimized = self.mtp_data.optimized
        return jac

    def jac_forces(self, atoms: Atoms) -> MTPData:
        """Calculate the Jacobian of the forces with respect to the MTP parameters.

        `jac.parameters` have the shape of `(nparams, natoms, 3)`.

        """
        spc = self.mtp_data.species_count
        number_of_atoms = len(atoms)

        jac = Jac(
            scaling=np.zeros((1, number_of_atoms, 3)),
            moment_coeffs=self.mbd.dbdris * -1.0,
            species_coeffs=np.zeros((spc, number_of_atoms, 3)),
            radial_coeffs=self.mbd.dgdcs * -1.0,
        )  # placeholder of the Jacobian with respect to the parameters
        jac.optimized = self.mtp_data.optimized
        return jac

    def jac_stress(self, atoms: Atoms) -> MTPData:
        """Calculate the Jacobian of the forces with respect to the MTP parameters.

        `jac.parameters` have the shape of `(nparams, natoms, 3)`.

        """
        spc = self.mtp_data.species_count

        jac = Jac(
            scaling=np.zeros((1, 3, 3)),
            moment_coeffs=self.mbd.dbdeps.copy(),
            species_coeffs=np.zeros((spc, 3, 3)),
            radial_coeffs=self.mbd.dsdcs.copy(),
        )  # placeholder of the Jacobian with respect to the parameters
        jac.optimized = self.mtp_data.optimized
        return jac


def _compute_offsets(nl: PrimitiveNeighborList, atoms: Atoms):
    cell = atoms.cell
    return [nl.get_neighbors(j)[1] @ cell for j in range(len(atoms))]


def _compute_all_offsets(nl: PrimitiveNeighborList, atoms: Atoms):
    cell = atoms.cell
    js = [nl.get_neighbors(i)[0] for i in range(len(atoms))]
    offsets = [nl.get_neighbors(i)[1] @ cell for i in range(len(atoms))]
    num_js = [_.shape[0] for _ in js]
    max_num_js = np.max([_.shape[0] for _ in offsets])
    pads = [(0, max_num_js - n) for n in num_js]
    # Pad dummy indices as -1 to recognize later
    padded_js = [
        np.pad(js_, pad_width=pad, constant_values=-1) for js_, pad in zip(js, pads)
    ]
    padded_offsets = [
        np.pad(offset, pad_width=(pad, (0, 0))) for offset, pad in zip(offsets, pads)
    ]
    return np.array(padded_js, dtype=int), np.array(padded_offsets)
