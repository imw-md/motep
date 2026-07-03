import logging
from abc import abstractmethod
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from ase import Atoms
from ase.neighborlist import PrimitiveNeighborList

from motep.potentials.mtp.data import MTPData

logger = logging.getLogger(__name__)


def _warn_if_neighbors_below_min_dist(
    neighbor_vectors: npt.NDArray[np.float64],
    min_dist: float,
) -> None:
    distances = np.linalg.norm(neighbor_vectors, axis=-1)
    min_distance = np.min(distances)
    if min_distance < min_dist:
        logger.warning(
            "Nearest-neighbor distance (%s A) is smaller than min_dist (%s A).",
            f"{min_distance:.6f}",
            f"{min_dist:.6f}",
        )


@dataclass
class MomentBasisData:
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
    dvdcs: np.ndarray
        Derivatives of local energies with respect to the radial coefficients.

    .. [Podryabinkin_CMS_2017_Active]
       E. V. Podryabinkin and A. V. Shapeev, Comput. Mater. Sci. 140, 171 (2017).

    """

    vatoms: npt.NDArray[np.float64] = field(default_factory=lambda: np.array(np.nan))
    dbdris: npt.NDArray[np.float64] = field(default_factory=lambda: np.array(np.nan))
    dbdeps: npt.NDArray[np.float64] = field(default_factory=lambda: np.array(np.nan))
    dvdcs: npt.NDArray[np.float64] = field(default_factory=lambda: np.array(np.nan))
    dgdcs: npt.NDArray[np.float64] = field(default_factory=lambda: np.array(np.nan))
    dsdcs: npt.NDArray[np.float64] = field(default_factory=lambda: np.array(np.nan))

    @property
    def values(self) -> np.ndarray:
        """Basis values summed over atoms."""
        return self.vatoms.sum(axis=-1)

    @property
    def dedcs(self) -> np.ndarray:
        """Derivatives of local energies with respect to the radial coefficients."""
        return self.dvdcs.sum(axis=-1)

    def initialize(self, natoms: int, mtp_data: MTPData, *, jac: bool) -> None:
        spc = mtp_data.species_count
        rfc = mtp_data.radial_funcs_count
        rbs = mtp_data.radial_basis.size
        asm = mtp_data.alpha_scalar_moments

        self.vatoms = np.full((asm, natoms), np.nan)
        if jac:
            self.dbdris = np.full((asm, natoms, 3), np.nan)
            self.dbdeps = np.full((asm, 3, 3), np.nan)
            self.dvdcs = np.full((spc, spc, rfc, rbs, natoms), np.nan)
            self.dgdcs = np.full((spc, spc, rfc, rbs, natoms, 3), np.nan)
            self.dsdcs = np.full((spc, spc, rfc, rbs, 3, 3), np.nan)

    def clean(self, *, jac: bool) -> None:
        self.vatoms[...] = 0.0
        if jac:
            self.dbdris[...] = 0.0
            self.dbdeps[...] = 0.0
            self.dvdcs[...] = 0.0
            self.dgdcs[...] = 0.0
            self.dsdcs[...] = 0.0


@dataclass
class RadialBasisData:
    """Data related to the radial basis.

    Attributes
    ----------
    values : np.ndarray (species_count, species_count, radial_basis.size)
        Radial basis values summed over atoms.
    dqdris : (species_count, species_count, radial_basis.size, 3, natoms)
        Derivaties of radial basis functions summed over atoms.

    """

    values: npt.NDArray[np.float64] = field(default_factory=lambda: np.array(np.nan))
    dqdris: npt.NDArray[np.float64] = field(default_factory=lambda: np.array(np.nan))
    dqdeps: npt.NDArray[np.float64] = field(default_factory=lambda: np.array(np.nan))

    def initialize(self, natoms: int, mtp_data: MTPData, *, jac: bool) -> None:
        spc = mtp_data.species_count
        rbs = mtp_data.radial_basis.size

        if jac:
            self.values = np.full((spc, spc, rbs), np.nan)
            self.dqdris = np.full((spc, spc, rbs, natoms, 3), np.nan)
            self.dqdeps = np.full((spc, spc, rbs, 3, 3), np.nan)

    def clean(self, *, jac: bool) -> None:
        if jac:
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


class EngineWithNeighborlist:
    """Class for getting interatomic vectors and hanling neighbor list."""

    static_geometry: bool = False

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
        With ``static_geometry`` the interatomic vectors are computed once and
        the neighborlist then discarded.

        """
        if self.static_geometry:
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
        if self.static_geometry and hasattr(self, "_interatomic_vectors"):
            return self._interatomic_vectors
        max_dist = self.mtp_data.radial_basis.max
        positions = atoms.positions
        offsets = self._offsets
        interatomic_vectors = positions[self._neighbors]  # r_j
        interatomic_vectors += offsets  # account for periodic images
        interatomic_vectors -= positions[:, None, :]  # r_i
        interatomic_vectors[self._neighbors[:, :] < 0, :] = max_dist
        valid_neighbors = self._neighbors >= 0
        neighbor_vectors = interatomic_vectors[valid_neighbors]
        if neighbor_vectors.size:
            _warn_if_neighbors_below_min_dist(
                neighbor_vectors,
                self.mtp_data.radial_basis.min,
            )
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
        static_geometry: bool = False,
    ) -> None:
        """MLIP-2 MTP.

        Parameters
        ----------
        mtp_data : :class:`motep.potentials.mtp.data.MTPData`
            Parameters in the MLIP .mtp file.
        static_geometry : bool, default False
            If True, the atomic geometry is assumed fixed across calls and the
            interatomic vectors are cached once (suitable for training, where
            only parameters change).

        """
        self._last_state: tuple | None = None
        self.update(mtp_data)
        self.results = {}
        self.neighbor_list = None
        self.static_geometry = static_geometry
        self._jac_ready = False

        self.mbd = MomentBasisData()
        self.rbd = RadialBasisData()

    def update(self, mtp_data: MTPData) -> bool:
        """Update MTP parameters.

        Returns
        -------
        bool
            Whether any input that determines the computed energy/forces
            changed since the previous update (see
            :meth:`MTPData.basis_state`). When ``False``, the cached results may
            be reused without recomputing the basis.

        """
        self.mtp_data = mtp_data
        if self.mtp_data.species.size == 0:
            self.mtp_data.species = list(range(self.mtp_data.species_count))
        state = mtp_data.basis_state()
        changed = not mtp_data.basis_state_equal(state, self._last_state)
        self._last_state = state
        return changed

    def check_species(self, atoms: Atoms) -> None:
        """Check if `atoms` comply with the `mtp_data.species`.

        Raises
        ------
        ValueError
            If the unique `atoms.numbers` is larger than `species_count`, or if
            they are not all present in `mtp_data.species`.

        """
        unique_species = np.unique(atoms.numbers)
        if unique_species.size > self.mtp_data.species_count:
            msg = "The number of species in input atoms is larger than species_count."
            raise ValueError(msg)
        if not all(_ in self.mtp_data.species for _ in unique_species):
            msg = (
                "All species in input atoms are not in mtp_data.species.\n"
                f"  species in atoms: {unique_species}\n"
                f"  species in mtp_data: {self.mtp_data.species}"
            )
            raise ValueError(msg)

    def initialize_basis_data(self, atoms: Atoms, *, jac: bool) -> None:
        """(Re)initialize moment and radial basis data."""
        natoms = len(atoms)
        self.mbd.initialize(natoms, self.mtp_data, jac=jac)
        self.rbd.initialize(natoms, self.mtp_data, jac=jac)
        self._jac_ready = jac

    @abstractmethod
    def _calculate(self, atoms: Atoms, *, jac: bool) -> tuple: ...

    def efs(self, atoms: Atoms) -> dict:
        """Compute energies, forces, and stress."""
        return self._run(atoms, jac=False)

    def jac(self, atoms: Atoms) -> dict:
        """Compute energies, forces, stress and populate the parameter Jacobian.

        The per-target Jacobians are then available via ``jac_energy`` etc.
        """
        return self._run(atoms, jac=True)

    def _run(self, atoms: Atoms, *, jac: bool) -> dict:
        self.check_species(atoms)
        if self.update_neighbor_list(atoms) or (jac and not self._jac_ready):
            self.initialize_basis_data(atoms, jac=jac)

        energies, forces, stress = self._calculate(atoms, jac=jac)

        self._symmetrize_stress(atoms, stress, jac=jac)

        self.results = {}
        self.results["energies"] = energies
        self.results["energy"] = self.results["energies"].sum()
        self.results["forces"] = forces
        self.results["stress"] = stress.flat[[0, 4, 8, 5, 2, 1]]

        return self.results

    def _symmetrize_stress(self, atoms: Atoms, stress: np.ndarray, *, jac: bool) -> None:
        if atoms.cell.rank == 3:  # noqa: PLR2004
            volume = atoms.get_volume()
            stress += stress.T
            stress *= 0.5 / volume
            if jac:
                self.mbd.dbdeps += self.mbd.dbdeps.transpose(0, 2, 1)
                self.mbd.dbdeps *= 0.5 / volume
                self.mbd.dsdcs += self.mbd.dsdcs.swapaxes(-2, -1)
                self.mbd.dsdcs *= 0.5 / volume
                axes = 0, 1, 2, 4, 3
                self.rbd.dqdeps += self.rbd.dqdeps.transpose(axes)
                self.rbd.dqdeps *= 0.5 / volume
        else:
            stress[:, :] = np.nan
            if jac:
                self.mbd.dbdeps[:, :, :] = np.nan
                self.rbd.dqdeps[:, :, :] = np.nan

    def jac_energy(self, atoms: Atoms) -> MTPData:
        """Calculate the Jacobian of the energy with respect to the MTP parameters.

        Returns
        -------
        MTPData
            Placeholder of the Jacobian of the energy with respect tothe MTP parameters.

        """
        sps = self.mtp_data.species
        nbs = list(atoms.numbers)

        return Jacobian(
            scaling=0.0,
            moment_coeffs=self.mbd.values.copy(),
            species_coeffs=np.fromiter((nbs.count(s) for s in sps), dtype=float),
            radial_coeffs=self.mbd.dedcs.copy(),
            optimized=self.mtp_data.optimized,
        )  # placeholder of the Jacobian with respect to the parameters

    def jac_energies(self, atoms: Atoms) -> Jacobian:
        """Calculate the Jacobian of local energies with respect to the MTP parameters.

        Returns
        -------
        Jacobian
            Jacobian whose ``parameters`` have the shape of `(nparams, natoms)`.

        """
        number_of_atoms = len(atoms)
        spicies_coeffs = np.full((self.mtp_data.species_count, number_of_atoms), np.nan)
        for i, s in enumerate(self.mtp_data.species):
            spicies_coeffs[i] = atoms.numbers == s

        return Jacobian(
            scaling=np.zeros((1, number_of_atoms)),
            moment_coeffs=self.mbd.vatoms,
            species_coeffs=spicies_coeffs,
            radial_coeffs=self.mbd.dvdcs,
            optimized=self.mtp_data.optimized,
        )

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
