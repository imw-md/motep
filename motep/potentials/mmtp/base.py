from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt
from ase import Atoms

from motep.potentials.mtp.base import (
    EngineBase,
    Jac,
    ModeBase,
    MomentBasisData,
    RadialBasisData,
)

from .data import MagMTPData


class MagModeBase(ModeBase):
    """Base mode class for magnetic MTPs defining available operation modes."""

    all_modes: ClassVar[list[str]] = ["run", "train", "train_mgrad"]


@dataclass
class MagMomentBasisData(MagModeBase, MomentBasisData):
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

    dgmdcs: npt.NDArray[np.float64] | None = None

    def initialize(self, natoms: int, mtp_data: MagMTPData) -> None:
        """Initialize moment basis properties."""
        spc = mtp_data.species_count
        rfc = mtp_data.radial_funcs_count
        nrb = mtp_data.radial_basis_size * mtp_data.mag_basis_size**2
        asm = mtp_data.alpha_scalar_moments

        self.vatoms = np.full((asm, natoms), np.nan)
        if "train" in self.mode:
            self.dbdris = np.full((asm, natoms, 3), np.nan)
            self.dbdeps = np.full((asm, 3, 3), np.nan)
            self.dedcs = np.full((spc, spc, rfc, nrb), np.nan)
            self.dgdcs = np.full((spc, spc, rfc, nrb, natoms, 3), np.nan)
            self.dsdcs = np.full((spc, spc, rfc, nrb, 3, 3), np.nan)

        if self.mode == "train_mgrad":
            self.dbdmis = np.full((asm, natoms), np.nan)
            self.dgmdcs = np.full((spc, spc, rfc, nrb, natoms), np.nan)

    def clean(self) -> None:
        """Clean up moment basis properties."""
        super().clean()
        if self.mode == "train_mgrad":
            self.dbdmis[...] = 0.0
            self.dgmdcs[...] = 0.0


@dataclass
class MagRadialBasisData(MagModeBase, RadialBasisData):
    """Data related to the radial basis.

    Attributes
    ----------
    values : np.ndarray (species_count, species_count, nrb)
        Radial basis values summed over atoms.
    dqdris : (species_count, species_count, nrb, 3, natoms)
        Derivaties of radial basis functions summed over atoms.

    Notes
    -----
    `nrb` is the combined radial basis size, which is the product of
    `radial_basis_size` and `mag_basis_size**2`.
    """

    dqdmis: npt.NDArray[np.float64] | None = None

    def initialize(self, natoms: int, mtp_data: MagMTPData) -> None:
        """Initialize radial basis properties."""
        spc = mtp_data.species_count
        nrb = mtp_data.radial_basis_size * mtp_data.mag_basis_size**2

        if "train" in self.mode:
            self.values = np.full((spc, spc, nrb), np.nan)
            self.dqdris = np.full((spc, spc, nrb, natoms, 3), np.nan)
            self.dqdeps = np.full((spc, spc, nrb, 3, 3), np.nan)

        if self.mode == "train_mgrad":
            self.dqdmis = np.full((spc, spc, nrb, natoms), np.nan)

    def clean(self) -> None:
        """Clean up radial basis properties."""
        super().clean()
        if self.mode == "train_mgrad":
            self.dqdmis[...] = 0.0


class MagEngineBase(MagModeBase, EngineBase):
    """Engine to compute an MTP."""

    def __init__(
        self,
        mtp_data: MagMTPData,
        *,
        mode: str = "run",
    ) -> None:
        """MLIP-2 MTP.

        Parameters
        ----------
        mtp_data : :class:`motep.potentials.mtp.data.MTPData`
            Parameters in the MLIP .mtp file.
        mode : {'run', 'train', 'train_mgrad'}, default 'run'
            Mode of operation. 'train' computes and stores basis data for
            training; 'train_mgrad' additionally computes basis data for
            training to magnetic gradients; 'run' is the default runtime mode.

        """
        self.update(MagMTPData.from_base(mtp_data))
        self.results = {}
        self.neighbor_list = None
        self.mode = mode

        # moment basis data
        self.mbd = MagMomentBasisData(mode=mode)

        # used for `Level2MTPOptimizer`
        self.rbd = MagRadialBasisData(mode=mode)

    def update(self, mtp_data: MagMTPData) -> None:
        """Update MTP parameters."""
        self.mtp_data: MagMTPData = mtp_data
        if self.mtp_data.species is None:
            self.mtp_data.species = list(range(self.mtp_data.species_count))

    def calculate(self, atoms: Atoms) -> dict:
        """Calculate properties of the given system.

        Returns
        -------
        Dictionary with energies, energy, forces and stress.

        """
        self.check_species(atoms)
        if self.update_neighbor_list(atoms):
            self.initialize_basis_data(atoms)

        energies, forces, stress, mgrad = self._calculate(atoms)

        self._symmetrize_stress(atoms, stress)

        self.results["energies"] = energies
        self.results["energy"] = self.results["energies"].sum()
        self.results["forces"] = forces
        self.results["stress"] = stress.flat[[0, 4, 8, 5, 2, 1]]
        self.results["mgrad"] = mgrad

        return self.results

    def jac_mgrad(self, atoms: Atoms) -> Jac:
        """Calculate the Jacobian of the magnetic gradient with respect to the MTP parameters.

        `jac.parameters` have the shape of `(nparams, natoms)`.

        """
        spc = self.mtp_data.species_count
        number_of_atoms = len(atoms)

        jac = Jac(
            scaling=np.zeros((1, number_of_atoms)),
            moment_coeffs=self.mbd.dbdmis,
            species_coeffs=np.zeros((spc, number_of_atoms)),
            radial_coeffs=self.mbd.dgmdcs,
        )  # placeholder of the Jacobian with respect to the parameters
        jac.optimized = self.mtp_data.optimized
        return jac
