from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt
from ase import Atoms
from scipy.optimize import Bounds, minimize

from motep.potentials.mtp.base import (
    EngineBase,
    Jacobian,
    ModeBase,
    MomentBasisData,
    RadialBasisData,
)

from .data import MagMTPData


def _ensure_collinear_magmoms(
    atoms: Atoms, magmoms: npt.NDArray[np.float64] | None = None
) -> npt.NDArray[np.float64]:
    """Ensure magnetic moments are a 1D collinear array.

    Parameters
    ----------
    atoms : Atoms
        The atomic structure.
    magmoms : np.ndarray or None
        Magnetic moments. If None, reads from
        ``atoms.get_initial_magnetic_moments()``.

    Returns
    -------
    1D array of scalar magnetic moments.

    Raises
    ------
    ValueError
        If 2D magmoms have more than one non-zero component (non-collinear).

    """
    if magmoms is None:
        magmoms = atoms.get_initial_magnetic_moments()
    if magmoms.ndim == 2:
        nonzero_cols = np.where(~(magmoms == 0).all(axis=0))[0]
        if len(nonzero_cols) > 1:
            msg = (
                "Non-collinear magnetic moments (multiple non-zero "
                "components) are not supported. Got non-zero columns: "
                f"{nonzero_cols.tolist()}"
            )
            raise ValueError(msg)
        if len(nonzero_cols) == 1:
            magmoms = magmoms[:, nonzero_cols[0]]
        else:
            magmoms = np.zeros(magmoms.shape[0])
    return np.asarray(magmoms, dtype=np.float64)


class MagModeBase(ModeBase):
    """Base mode class for magnetic MTPs defining available operation modes."""

    all_modes: ClassVar[list[str]] = ["run", "train", "train_mgrad"]


@dataclass
class MagMomentBasisData(MagModeBase, MomentBasisData):
    """Data related to the moment basis."""

    dgmdcs: npt.NDArray[np.float64] | None = None

    def initialize(self, natoms: int, mtp_data: MagMTPData) -> None:
        """Initialize moment basis properties."""
        spc = mtp_data.species_count
        rfc = mtp_data.radial_funcs_count
        nrb = mtp_data.radial_basis.size * mtp_data.magnetic_basis.size**2
        asm = mtp_data.alpha_scalar_moments

        self.vatoms = np.full((asm, natoms), np.nan)
        if "train" in self.mode:
            self.dbdris = np.full((asm, natoms, 3), np.nan)
            self.dbdeps = np.full((asm, 3, 3), np.nan)
            self.dvdcs = np.full((spc, spc, rfc, nrb, natoms), np.nan)
            self.dgdcs = np.full((spc, spc, rfc, nrb, natoms, 3), np.nan)
            self.dsdcs = np.full((spc, spc, rfc, nrb, 3, 3), np.nan)

        if self.mode == "train_mgrad":
            self.dbdmis = np.full((asm, natoms), np.nan)
            self.dgmdcs = np.full((spc, spc, rfc, nrb, natoms), np.nan)

    def clean(self, *, jac: bool = True, mgrad: bool = True) -> None:
        """Clean up moment basis properties."""
        super().clean(jac=jac)
        if mgrad and self.mode == "train_mgrad":
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
    `radial_basis.size` and `magnetic_basis.size**2`.
    """

    dqdmis: npt.NDArray[np.float64] | None = None

    def initialize(self, natoms: int, mtp_data: MagMTPData) -> None:
        """Initialize radial basis properties."""
        spc = mtp_data.species_count
        nrb = mtp_data.radial_basis.size * mtp_data.magnetic_basis.size**2

        if "train" in self.mode:
            self.values = np.full((spc, spc, nrb), np.nan)
            self.dqdris = np.full((spc, spc, nrb, natoms, 3), np.nan)
            self.dqdeps = np.full((spc, spc, nrb, 3, 3), np.nan)

        if self.mode == "train_mgrad":
            self.dqdmis = np.full((spc, spc, nrb, natoms), np.nan)

    def clean(self, *, jac: bool = True, mgrad: bool = True) -> None:
        """Clean up radial basis properties."""
        super().clean(jac=jac)
        if mgrad and self.mode == "train_mgrad":
            self.dqdmis[...] = 0.0


class MagEngineBase(MagModeBase, EngineBase):
    """Engine to compute an MTP."""

    def __init__(
        self,
        mtp_data: MagMTPData,
        *,
        mode: str = "run",
        static_geometry: bool = False,
    ) -> None:
        """Magnetic MTP as described in [Novikov_nCM_2022_Magnetic]_.

        Parameters
        ----------
        mtp_data : :class:`motep.potentials.mtp.data.MTPData`
            Parameters in the MLIP .mtp file.
        mode : {'run', 'train', 'train_mgrad'}, default 'run'
            Capability of the engine. ``'train'`` allocates the parameter-
            Jacobian basis data (``jac`` permitted); ``'train_mgrad'`` also
            allocates the magnetic-gradient Jacobian (``jac(mgrad=True)``);
            ``'run'`` allocates only what ``efs`` needs.
        static_geometry : bool, default False
            If True, the atomic geometry is assumed fixed across calls.

        .. [Novikov_nCM_2022_Magnetic]
            I. Novikov, Blazej Grabowski, Fritz Körmann and A. V. Shapeev, npj Comput. Mater. 8, 13 (2022).

        """
        self._last_state: tuple | None = None
        self.update(MagMTPData.from_base(mtp_data))
        self.results = {}
        self.neighbor_list = None
        self.mode = mode
        self.static_geometry = static_geometry
        self._jac_valid = False
        self._mgrad_valid = False

        # moment basis data
        self.mbd = MagMomentBasisData(mode=mode)

        # used for `Level2MTPOptimizer`
        self.rbd = MagRadialBasisData(mode=mode)

    def _require_train(self, *, mgrad: bool = False) -> None:
        if "train" not in self.mode:
            msg = f"jac() requires a train-mode engine, got mode={self.mode!r}."
            raise RuntimeError(msg)
        if mgrad and self.mode != "train_mgrad":
            msg = (
                "jac(mgrad=True) requires a 'train_mgrad' engine, got "
                f"mode={self.mode!r}."
            )
            raise RuntimeError(msg)

    def _require_mgrad(self) -> None:
        if not self._mgrad_valid:
            msg = (
                "magnetic-gradient Jacobian data is stale; call jac(mgrad=True) "
                "before reading it."
            )
            raise RuntimeError(msg)

    def get_mbd(self, *, mgrad: bool = False) -> "MagMomentBasisData":
        """Return the moment basis data, asserting the Jacobian is current."""
        mbd = super().get_mbd()
        if mgrad:
            self._require_mgrad()
        return mbd

    def get_rbd(self, *, mgrad: bool = False) -> "MagRadialBasisData":
        """Return the radial basis data, asserting the Jacobian is current."""
        rbd = super().get_rbd()
        if mgrad:
            self._require_mgrad()
        return rbd

    def update(self, mtp_data: MagMTPData) -> bool:
        """Update MTP parameters.

        Returns
        -------
        bool
            Whether the coefficients changed since the previous update.

        """
        self.mtp_data: MagMTPData = mtp_data
        changed = super().update(mtp_data)
        if changed:  # stored magnetic-gradient Jacobian no longer matches params
            self._mgrad_valid = False
        return changed

    def efs(self, atoms: Atoms, magmoms: npt.NDArray[np.float64] | None = None) -> dict:
        """Compute energies, forces, stress and the magnetic gradient."""
        return self._run_mag(atoms, magmoms, jac=False, mgrad=False)

    def jac(
        self,
        atoms: Atoms,
        magmoms: npt.NDArray[np.float64] | None = None,
        *,
        mgrad: bool = False,
    ) -> dict:
        """Compute results and populate the parameter-Jacobian basis data.

        With ``mgrad`` the magnetic-gradient Jacobian is populated as well
        (requires a ``train_mgrad`` engine).
        """
        self._require_train(mgrad=mgrad)
        return self._run_mag(atoms, magmoms, jac=True, mgrad=mgrad)

    def _run_mag(
        self,
        atoms: Atoms,
        magmoms: npt.NDArray[np.float64] | None,
        *,
        jac: bool,
        mgrad: bool,
    ) -> dict:
        self.check_species(atoms)
        if self.update_neighbor_list(atoms):
            self.initialize_basis_data(atoms)

        magmoms = _ensure_collinear_magmoms(atoms, magmoms)

        energies, forces, stress, mg = self._calculate(
            atoms, magmoms, jac=jac, mgrad=mgrad
        )

        self._symmetrize_stress(atoms, stress, jac=jac)

        self.results = {}
        self.results["energies"] = energies
        self.results["energy"] = self.results["energies"].sum()
        self.results["forces"] = forces
        self.results["stress"] = stress.flat[[0, 4, 8, 5, 2, 1]]
        self.results["mgrad"] = mg

        self._jac_valid = jac
        self._mgrad_valid = jac and mgrad
        return self.results

    def jac_mgrad(self, atoms: Atoms) -> Jacobian:
        """Calculate the Jacobian of the magnetic gradient with respect to parameters.

        `jac.parameters` have the shape of `(nparams, natoms)`.

        """
        self._require_mgrad()
        spc = self.mtp_data.species_count
        number_of_atoms = len(atoms)

        jac = Jacobian(
            scaling=np.zeros((1, number_of_atoms)),
            moment_coeffs=self.mbd.dbdmis,
            species_coeffs=np.zeros((spc, number_of_atoms)),
            radial_coeffs=self.mbd.dgmdcs,
            optimized=self.mtp_data.optimized,
        )  # placeholder of the Jacobian with respect to the parameters
        return jac

    def relax_magnetic_moments(
        self,
        atoms: Atoms,
        magmoms_init: npt.NDArray[np.float64] | None = None,
    ) -> dict:
        """Relax magnetic moments to minimize energy.

        Parameters
        ----------
        atoms : Atoms
            The atomic structure with magnetic moments to relax.
        magmoms_init : np.ndarray or None
            Initial guess for magnetic moments. If None, reads from
            ``atoms.get_initial_magnetic_moments()``.

        Returns
        -------
        dict
            Results dictionary including relaxed ``magmoms`` and ``magmom``.

        """
        magmoms_init = _ensure_collinear_magmoms(atoms, magmoms_init)
        mtp_data = self.mtp_data

        self.check_species(atoms)
        if self.update_neighbor_list(atoms):
            self.initialize_basis_data(atoms)

        def objective_and_grad(moms: np.ndarray) -> tuple[float, np.ndarray]:
            energies, _, _, mgrad = self._calculate(
                atoms,
                moms,
                jac=False,
                mgrad=False,
            )
            return float(energies.sum()), mgrad

        min_mag = mtp_data.magnetic_basis.min
        max_mag = mtp_data.magnetic_basis.max
        bounds = Bounds(np.full((len(atoms)), min_mag), np.full((len(atoms)), max_mag))
        result = minimize(
            objective_and_grad,
            magmoms_init,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
        )
        relaxed_magmoms = result.x
        results = self.efs(atoms, relaxed_magmoms)
        results["magmoms"] = relaxed_magmoms
        results["magmom"] = relaxed_magmoms.sum()
        return results
