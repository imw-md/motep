"""ASE Calculators."""

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from motep.potentials.mmtp.base import MagEngineBase
from motep.potentials.mmtp.data import MagMTPData
from motep.potentials.mtp.base import EngineBase
from motep.potentials.mtp.data import MTPData


def make_mtp_engine(engine: str = "cext") -> type[EngineBase]:
    """Make the MTP engine.

    Returns
    -------
    type[EngineBase]

    Raises
    ------
    ValueError

    """
    if engine == "numpy":
        from motep.potentials.mtp.numpy.engine import NumpyMTPEngine

        return NumpyMTPEngine
    if engine == "numba":
        from motep.potentials.mtp.numba.engine import NumbaMTPEngine

        return NumbaMTPEngine
    if engine == "numba_mag":
        from motep.potentials.mmtp.numba.engine import NumbaMagMTPEngine

        return NumbaMagMTPEngine
    if engine == "jax":
        from motep.potentials.mtp.jax.engine import JaxMTPEngine

        return JaxMTPEngine
    if engine == "cext":
        from motep.potentials.mtp.cext.engine import CExtMTPEngine

        return CExtMTPEngine
    if engine == "cext_mag":
        from motep.potentials.mmtp.cext.engine import CExtMagMTPEngine

        return CExtMagMTPEngine

    raise ValueError(engine)


class MTP(Calculator):
    """ASE Calculator of the MTP potential."""

    implemented_properties = (
        "energy",
        "free_energy",
        "energies",
        "forces",
        "stress",
    )

    def __init__(
        self,
        mtp_data: MTPData,
        *args: tuple,
        engine: str = "cext",
        mode: str = "run",
        **kwargs: dict,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.engine: EngineBase = make_mtp_engine(engine)(mtp_data, mode=mode)

    def update_parameters(self, mtp_data: MTPData) -> None:
        """Update MTP parameters."""
        self.engine.update(mtp_data)
        self.results = {}  # trigger new calculation

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] = ["energy"],
        system_changes: list[str] = all_changes,
    ) -> None:
        """Calculate."""
        super().calculate(atoms, properties, system_changes)

        self.results = self.engine.calculate(self.atoms)

        self.results["free_energy"] = self.results["energy"]

        if self.atoms.cell.rank != 3 and "stress" in self.results:
            del self.results["stress"]


class MMTP(MTP):
    """ASE Calculator of the magnetic MTP potential.

    Parameters
    ----------
    mtp_data : MagMTPData
        Magnetic MTP potential data.
    engine : str
        Engine backend to use.
    mode : str
        Operation mode ('run', 'train', 'train_mgrad').
    relax_magmoms : bool
        If True, magnetic moments are relaxed to minimize energy before
        returning EFS results. If False (default), EFS is computed at the
        given initial magnetic moments (fixed-magmom mode for training).
    warm_start_magmoms : bool
        If True (default) and ``relax_magmoms=True``, the relaxation is
        warm-started from the previously relaxed magnetic moments when only
        positions/cell changed. Set to False to always start from
        ``atoms.get_initial_magnetic_moments()``. Ignored when
        ``relax_magmoms=False``.

    """

    implemented_properties = (
        "energy",
        "free_energy",
        "energies",
        "forces",
        "stress",
        "magmom",
        "magmoms",
    )

    default_parameters = {
        "relax_magmoms": False,
        "warm_start_magmoms": True,
    }

    def __init__(
        self,
        mtp_data: MagMTPData,
        *args,
        engine: str = "cext_mag",
        mode: str = "run",
        **kwargs,
    ) -> None:
        Calculator.__init__(self, *args, **kwargs)
        self.engine: MagEngineBase = make_mtp_engine(engine)(mtp_data, mode=mode)
        self._relaxed_magmoms: np.ndarray | None = None

    def update_parameters(self, mtp_data: MagMTPData) -> None:
        self.engine.update(mtp_data)
        self.results = {}  # trigger new calculation

    def reset(self) -> None:
        """Clear all information from old calculation."""
        super().reset()
        self._relaxed_magmoms = None

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] = ["energy"],
        system_changes: list[str] = all_changes,
    ) -> None:
        Calculator.calculate(self, atoms, properties, system_changes)

        relax = self.parameters.get("relax_magmoms", False)

        if relax:
            # Determine initial guess for relaxation
            if "initial_magmoms" in system_changes:
                # Initial magmoms changed — different basin, start fresh
                self._relaxed_magmoms = None

            warm_start = self.parameters.get("warm_start_magmoms", True)
            if warm_start and self._relaxed_magmoms is not None:
                magmoms_init = self._relaxed_magmoms
            else:
                magmoms_init = None  # engine will read from atoms

            self.results = self.engine.relax_magnetic_moments(
                self.atoms, magmoms_init=magmoms_init
            )
            self._relaxed_magmoms = self.results["magmoms"].copy()
        else:
            self.results = self.engine.calculate(self.atoms)

        self.results["free_energy"] = self.results["energy"]

        if self.atoms.cell.rank != 3 and "stress" in self.results:
            del self.results["stress"]
