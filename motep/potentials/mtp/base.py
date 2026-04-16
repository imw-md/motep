from abc import abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import numpy.typing as npt
from ase import Atoms
from ase.neighborlist import PrimitiveNeighborList

from motep.potentials.mtp.data import MTPData


class ModeBase:
    """Base class for operating modes."""

    all_modes: ClassVar[list[str]] = ["run", "train"]

    @property
    def mode(self) -> str:
        """Get the current operating mode."""
        return self._mode

    @mode.setter
    def mode(self, mode: str) -> None:
        if mode not in self.all_modes:
            raise NotImplementedError(mode)
        self._mode = mode


@dataclass
class MomentBasisData(ModeBase):
    """Data related to the moment basis.

    Attributes
    ----------
    values : np.ndarray (alpha_moments_count)
        Basis values summed over atoms.
        This corresponds to b_j in Eq. (5) in [Podryabinkin_CMS_2017_Active]_.
    vatoms : np.ndarray (alpha_moments_count, number_of_atoms)
        Basis values per atom.
        This corresponds to B_j in Eq. (2) in [Podryabinkin_CMS_2017_Active]_.
    dbdris : np.ndarray (alpha_moments_count, number_of_atoms, 3)
        Derivatives of basis functions with respect to Cartesian coordinates of atoms
        summed over atoms.
        This corresponds to nabla b_j in Eq. (7a) in [Podryabinkin_CMS_2017_Active]_.
    dbdeps : np.ndarray (alpha_moments_count, 3, 3)
        Derivatives of cumulated basis functions with respect to the strain tensor.

    .. [Podryabinkin_CMS_2017_Active]
       E. V. Podryabinkin and A. V. Shapeev, Comput. Mater. Sci. 140, 171 (2017).

    """

    mode: str = field(default="run")

    vatoms: npt.NDArray[np.float64] = field(default_factory=lambda: np.array(np.nan))
    dbdris: npt.NDArray[np.float64] = field(default_factory=lambda: np.array(np.nan))
    dbdeps: npt.NDArray[np.float64] = field(default_factory=lambda: np.array(np.nan))
    dedcs: npt.NDArray[np.float64] = field(default_factory=lambda: np.array(np.nan))
    dgdcs: npt.NDArray[np.float64] = field(default_factory=lambda: np.array(np.nan))
    dsdcs: npt.NDArray[np.float64] = field(default_factory=lambda: np.array(np.nan))

    @property
    def values(self) -> np.ndarray:
        """Basis values summed over atoms."""
        return self.vatoms.sum(axis=-1)

    def initialize(self, natoms: int, mtp_data: MTPData) -> None:
        """Initialize moment basis properties."""
        spc = mtp_data.species_count
        rfc = mtp_data.radial_funcs_count
        rbs = mtp_data.radial_basis.size
        asm = mtp_data.alpha_scalar_moments

        self.vatoms = np.full((asm, natoms), np.nan)
        if "train" in self.mode:
            self.dbdris = np.full((asm, natoms, 3), np.nan)
            self.dbdeps = np.full((asm, 3, 3), np.nan)
            self.dedcs = np.full((spc, spc, rfc, rbs), np.nan)
            self.dgdcs = np.full((spc, spc, rfc, rbs, natoms, 3), np.nan)
            self.dsdcs = np.full((spc, spc, rfc, rbs, 3, 3), np.nan)

    def clean(self) -> None:
        """Clean up moment basis properties."""
        self.vatoms[...] = 0.0
        if "train" in self.mode:
            self.dbdris[...] = 0.0
            self.dbdeps[...] = 0.0
            self.dedcs[...] = 0.0
            self.dgdcs[...] = 0.0
            self.dsdcs[...] = 0.0


@dataclass
class RadialBasisData(ModeBase):
    """Data related to the radial basis.

    Attributes
    ----------
    values : np.ndarray (species_count, species_count, radial_basis.size)
        Radial basis values summed over atoms.
    dqdris : (species_count, species_count, radial_basis.size, 3, natoms)
        Derivaties of radial basis functions summed over atoms.

    """

    mode: str = field(default="run")

    values: npt.NDArray[np.float64] = field(default_factory=lambda: np.array(np.nan))
    dqdris: npt.NDArray[np.float64] = field(default_factory=lambda: np.array(np.nan))
    dqdeps: npt.NDArray[np.float64] = field(default_factory=lambda: np.array(np.nan))

    def initialize(self, natoms: int, mtp_data: MTPData) -> None:
        """Initialize radial basis properties."""
        spc = mtp_data.species_count
        rbs = mtp_data.radial_basis.size

        if "train" in self.mode:
            self.values = np.full((spc, spc, rbs), np.nan)
            self.dqdris = np.full((spc, spc, rbs, natoms, 3), np.nan)
            self.dqdeps = np.full((spc, spc, rbs, 3, 3), np.nan)

    def clean(self) -> None:
        """Clean up radial basis properties."""
        if "train" in self.mode:
            self.values[...] = 0.0
            self.dqdris[...] = 0.0
            self.dqdeps[...] = 0.0


@dataclass(kw_only=True)
class Jacobian:
    """Placeholder of the Jacobian with respect to the MTP parameters."""

    scaling: npt.NDArray[np.float64]
    radial_coeffs: npt.NDArray[np.float64]
    species_coeffs: npt.NDArray[np.float64]
    moment_coeffs: npt.NDArray[np.float64]
    optimized: list[str]

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


class EngineWithNeighborlist(ModeBase):
    """Class for getting interatomic vectors and hanling neighbor list."""

    def __init__(self) -> None:
        self.neighbor_list: PrimitiveNeighborList | None = None
        self.mtp_data: MTPData = MTPData()

    def _initiate_neighbor_list(self, atoms: Atoms) -> None:
        """Initialize the ASE `PrimitiveNeighborList` object."""
        self.neighbor_list = PrimitiveNeighborList(
            cutoffs=[0.5 * self.mtp_data.radial_basis.max] * len(atoms),
            skin=0.3,  # cutoff + skin is used, recalc only if diff in pos > skin
            self_interaction=False,  # Exclude [0, 0, 0]
            bothways=True,  # return both ij and ji
        )
        self.neighbor_list.update(atoms.pbc, atoms.cell, atoms.positions)
        self._neighbors, self._offsets = self._get_neighbors_and_offsets(atoms)

    def update_neighbor_list(self, atoms: Atoms) -> None:
        """Update the ASE `PrimitiveNeighborList` object.

        Notes
        -----
        If in training mode, only the interatomic vectors are computed once, and
        the neighborlist then discarded.

        """
        # Special handling if in train mode
        if "train" in self.mode:
            if not hasattr(self, "_interatomic_vectors"):
                self._initiate_neighbor_list(atoms)
                self._interatomic_vectors = self._get_interatomic_vectors(atoms)
                self.neighbor_list = None
                del self._offsets
                return True
            return False

        if self.neighbor_list is None:
            self._initiate_neighbor_list(atoms)
            return True
        if self.neighbor_list.update(atoms.pbc, atoms.cell, atoms.positions):
            self._neighbors, self._offsets = self._get_neighbors_and_offsets(atoms)
            return True
        return False

    def _get_interatomic_vectors(self, atoms: Atoms) -> np.ndarray:
        if "train" in self.mode and hasattr(self, "_interatomic_vectors"):
            return self._interatomic_vectors
        max_dist = self.mtp_data.radial_basis.max
        positions = atoms.positions
        offsets = self._offsets
        interatomic_vectors = positions[self._neighbors]  # r_j
        interatomic_vectors += offsets  # account for periodic images
        interatomic_vectors -= positions[:, None, :]  # r_i
        interatomic_vectors[self._neighbors[:, :] < 0, :] = max_dist
        return interatomic_vectors

    def _get_neighbors_and_offsets(self, atoms: Atoms) -> tuple[np.ndarray, np.ndarray]:
        nl = self.neighbor_list
        cell = atoms.cell
        n_atoms = len(atoms)
        # First pass: find max_num_js
        max_num_js = 0
        for i in range(n_atoms):
            _, offsets_i = nl.get_neighbors(i)
            max_num_js = max(max_num_js, offsets_i.shape[0])

        # Preallocate arrays
        js = np.full((n_atoms, max_num_js), -1, dtype=np.int32)
        offsets = np.zeros((n_atoms, max_num_js, 3))

        for i in range(n_atoms):
            js_i, offsets_i = nl.get_neighbors(i)
            n_neighbors = js_i.shape[0]
            js[i, :n_neighbors] = js_i
            offsets[i, :n_neighbors] = offsets_i @ cell

        return js, offsets


class EngineBase(EngineWithNeighborlist):
    """Engine to compute an MTP."""

    def __init__(
        self,
        mtp_data: MTPData,
        *,
        mode: str = "run",
    ) -> None:
        """MLIP-2 MTP.

        Parameters
        ----------
        mtp_data : :class:`motep.potentials.mtp.data.MTPData`
            Parameters in the MLIP .mtp file.
        mode : {'run', 'train'}, default 'run'
            Mode of operation. `'train'` computes and stores basis data for
            training; `'run'` is the default mode suitable for MD.

        """
        self.update(mtp_data)
        self.results = {}
        self.neighbor_list = None
        self.mode = mode

        # moment basis data
        self.mbd = MomentBasisData(mode=mode)

        # used for `Level2MTPOptimizer`
        self.rbd = RadialBasisData(mode=mode)

    def update(self, mtp_data: MTPData) -> None:
        """Update MTP parameters."""
        self.mtp_data = mtp_data
        if self.mtp_data.species.size == 0:
            self.mtp_data.species = list(range(self.mtp_data.species_count))

    def check_species(self, atoms: Atoms) -> None:
        """Check if `atoms` comply with the `mtp_data.species`.

        Raises
        ------
        ValueError
            If the unique `atoms.numbers` is larger than `species_count`, or if
            they are not present in `mtp_data.species` (only when not in
            `'train'` mode).

        """
        if np.unique(atoms.numbers).size > self.mtp_data.species_count:
            msg = "The number of species in input atoms is larger than species_count."
            raise ValueError(msg)
        if self.mode != "train":
            unique_species = np.unique(atoms.numbers)
            if not all(_ in self.mtp_data.species for _ in unique_species):
                msg = (
                    "All species in input atoms are not in mtp_data.species.\n"
                    f"  species in atoms: {unique_species}\n"
                    f"  species in mtp_data: {self.mtp_data.species}"
                )
                raise ValueError(msg)

    def initialize_basis_data(self, atoms: Atoms) -> None:
        """(Re)initialize moment and radial basis data."""
        natoms = len(atoms)
        self.mbd.initialize(natoms, self.mtp_data)
        self.rbd.initialize(natoms, self.mtp_data)

    @abstractmethod
    def _calculate(self, atoms: Atoms) -> tuple: ...

    def calculate(self, atoms: Atoms) -> dict:
        """Calculate properties of the given system.

        Returns
        -------
        Dictionary with energies, energy, forces and stress.

        """
        self.check_species(atoms)
        if self.update_neighbor_list(atoms):
            self.initialize_basis_data(atoms)

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
            if "train" in self.mode:
                self.mbd.dbdeps += self.mbd.dbdeps.transpose(0, 2, 1)
                self.mbd.dbdeps *= 0.5 / volume
                self.mbd.dsdcs += self.mbd.dsdcs.swapaxes(-2, -1)
                self.mbd.dsdcs *= 0.5 / volume
                axes = 0, 1, 2, 4, 3
                self.rbd.dqdeps += self.rbd.dqdeps.transpose(axes)
                self.rbd.dqdeps *= 0.5 / volume
        else:
            stress[:, :] = np.nan
            if "train" in self.mode:
                self.mbd.dbdeps[:, :, :] = np.nan
                self.rbd.dqdeps[:, :, :] = np.nan

    def jac_energy(self, atoms: Atoms) -> MTPData:
        """Calculate the Jacobian of the energy with respect to the MTP parameters."""
        sps = self.mtp_data.species
        nbs = list(atoms.numbers)

        jac = Jacobian(
            scaling=0.0,
            moment_coeffs=self.mbd.values.copy(),
            species_coeffs=np.fromiter((nbs.count(s) for s in sps), dtype=float),
            radial_coeffs=self.mbd.dedcs.copy(),
        )  # placeholder of the Jacobian with respect to the parameters
        jac.optimized = self.mtp_data.optimized
        return jac

    def jac_forces(self, atoms: Atoms) -> Jacobian:
        """Calculate the Jacobian of the forces with respect to the MTP parameters.

        Returns
        -------
        Jacobian
            Jacobian whose ``parameters`` have the shape of `(nparams, natoms, 3)`.

        """
        spc = self.mtp_data.species_count
        number_of_atoms = len(atoms)

        return Jacobian(
            scaling=np.zeros((1, number_of_atoms, 3)),
            moment_coeffs=self.mbd.dbdris * -1.0,
            species_coeffs=np.zeros((spc, number_of_atoms, 3)),
            radial_coeffs=self.mbd.dgdcs * -1.0,
            optimized=self.mtp_data.optimized,
        )  # placeholder of the Jacobian with respect to the parameters

    def jac_stress(self, atoms: Atoms) -> Jacobian:
        """Calculate the Jacobian of the forces with respect to the MTP parameters.

        Returns
        -------
        _Jacobian
            Jacobian whose ``parameters`` have the shape of `(nparams, 3, 3)`.

        """
        spc = self.mtp_data.species_count

        return Jacobian(
            scaling=np.zeros((1, 3, 3)),
            moment_coeffs=self.mbd.dbdeps.copy(),
            species_coeffs=np.zeros((spc, 3, 3)),
            radial_coeffs=self.mbd.dsdcs.copy(),
            optimized=self.mtp_data.optimized,
        )  # placeholder of the Jacobian with respect to the parameters
