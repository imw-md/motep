from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from ase import Atoms
from scipy.optimize import Bounds, minimize

from motep.potentials.mtp.base import (
    EngineBase,
    Jacobian,
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


@dataclass
class MagMomentBasisData(MomentBasisData):
    """Data related to the moment basis."""

    dgmdcs: npt.NDArray[np.float64] | None = None

    def initialize(
        self,
        natoms: int,
        mtp_data: MagMTPData,
        *,
        jac: bool,
        mgrad: bool,
    ) -> None:
        spc = mtp_data.species_count
        rfc = mtp_data.radial_funcs_count
        nrb = mtp_data.radial_basis.size * mtp_data.magnetic_basis.size**2
        asm = mtp_data.alpha_scalar_moments

        self.vatoms = np.full((asm, natoms), np.nan)
        if jac:
            self.dbdris = np.full((asm, natoms, 3), np.nan)
            self.dbdeps = np.full((asm, 3, 3), np.nan)
            self.dvdcs = np.full((spc, spc, rfc, nrb, natoms), np.nan)
            self.dgdcs = np.full((spc, spc, rfc, nrb, natoms, 3), np.nan)
            self.dsdcs = np.full((spc, spc, rfc, nrb, 3, 3), np.nan)

        if mgrad:
            self.dbdmis = np.full((asm, natoms), np.nan)
            self.dgmdcs = np.full((spc, spc, rfc, nrb, natoms), np.nan)

    def clean(self, *, jac: bool, mgrad: bool) -> None:
        super().clean(jac=jac)
        if mgrad:
            self.dbdmis[...] = 0.0
            self.dgmdcs[...] = 0.0


@dataclass
class MagRadialBasisData(RadialBasisData):
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

    def initialize(
        self, natoms: int, mtp_data: MagMTPData, *, jac: bool, mgrad: bool
    ) -> None:
        spc = mtp_data.species_count
        nrb = mtp_data.radial_basis.size * mtp_data.magnetic_basis.size**2

        if jac:
            self.values = np.full((spc, spc, nrb), np.nan)
            self.dqdris = np.full((spc, spc, nrb, natoms, 3), np.nan)
            self.dqdeps = np.full((spc, spc, nrb, 3, 3), np.nan)

        if mgrad:
            self.dqdmis = np.full((spc, spc, nrb, natoms), np.nan)

    def clean(self, *, jac: bool, mgrad: bool) -> None:
        super().clean(jac=jac)
        if mgrad:
            self.dqdmis[...] = 0.0


class MagEngineBase(EngineBase):
    """Engine to compute an MTP."""

    def __init__(
        self,
        mtp_data: MagMTPData,
        *,
        static_geometry: bool = False,
    ) -> None:
        """Magnetic MTP as described in [Novikov_nCM_2022_Magnetic]_.

        Parameters
        ----------
        mtp_data : :class:`motep.potentials.mtp.data.MTPData`
            Parameters in the MLIP .mtp file.
        static_geometry : bool, default False
            If True, the atomic geometry is assumed fixed across calls.

        .. [Novikov_nCM_2022_Magnetic]
            I. Novikov, Blazej Grabowski, Fritz Körmann and A. V. Shapeev, npj Comput. Mater. 8, 13 (2022).

        """
        self._last_state: tuple | None = None
        self.update(MagMTPData.from_base(mtp_data))
        self.results = {}
        self.neighbor_list = None
        self.static_geometry = static_geometry
        self._jac_ready = False
        self._mgrad_ready = False

        self.mbd = MagMomentBasisData()
        self.rbd = MagRadialBasisData()

    def update(self, mtp_data: MagMTPData) -> bool:
        """Update MTP parameters.

        Returns
        -------
        bool
            Whether the coefficients changed since the previous update.

        """
        self.mtp_data: MagMTPData = mtp_data
        return super().update(mtp_data)

    def initialize_basis_data(
        self, atoms: Atoms, *, jac: bool, mgrad: bool = False
    ) -> None:
        """(Re)initialize moment and radial basis data."""
        natoms = len(atoms)
        self.mbd.initialize(natoms, self.mtp_data, jac=jac, mgrad=mgrad)
        self.rbd.initialize(natoms, self.mtp_data, jac=jac, mgrad=mgrad)
        self._jac_ready = jac
        self._mgrad_ready = mgrad

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
        """Compute results and populate the parameter Jacobian.

        With ``mgrad`` the magnetic-gradient Jacobian is populated as well.
        """
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
        if (
            self.update_neighbor_list(atoms)
            or (jac and not self._jac_ready)
            or (mgrad and not self._mgrad_ready)
        ):
            self.initialize_basis_data(atoms, jac=jac, mgrad=mgrad)

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

        return self.results

    def jac_mgrad(self, atoms: Atoms) -> Jacobian:
        """Calculate the Jacobian of the magnetic gradient with respect to parameters.

        `jac.parameters` have the shape of `(nparams, natoms)`.

        """
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
            self.initialize_basis_data(atoms, jac=False, mgrad=False)

        def objective_and_grad(moms: np.ndarray) -> tuple[float, np.ndarray]:
            energies, _, _, mgrad = self._calculate(atoms, moms, jac=False, mgrad=False)
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
