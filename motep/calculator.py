"""ASE Calculators."""

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from motep.potentials.mmtp.base import MagEngineBase
from motep.potentials.mmtp.data import MagMTPData
from motep.potentials.mtp.base import EngineBase
from motep.potentials.mtp.data import MTPData

_ENGINES = {
    "numpy": "motep.potentials.mtp.numpy.engine.NumpyMTPEngine",
    "numba": "motep.potentials.mtp.numba.engine.NumbaMTPEngine",
    "jax": "motep.potentials.mtp.jax.engine.JaxMTPEngine",
    "cext": "motep.potentials.mtp.cext.engine.CExtMTPEngine",
}

_MAGNETIC_ENGINES = {
    "numba": "motep.potentials.mmtp.numba.engine.NumbaMagMTPEngine",
    "cext": "motep.potentials.mmtp.cext.engine.CExtMagMTPEngine",
}


def make_mtp_engine(
    engine: str = "cext",
    *,
    magnetic: bool = False,
) -> type[EngineBase]:
    """Return the engine class for the given backend name.

    Parameters
    ----------
    engine : str
        Backend name: ``"cext"``, ``"numba"``, ``"numpy"``, or ``"jax"``.
    magnetic : bool
        If True, return the magnetic variant of the engine.

    """
    import importlib

    registry = _MAGNETIC_ENGINES if magnetic else _ENGINES
    if engine not in registry:
        if magnetic:
            raise ValueError(
                f"Engine {engine!r} does not support magnetic potentials. "
                f"Supported: {sorted(_MAGNETIC_ENGINES)}"
            )
        raise ValueError(f"Unknown engine {engine!r}. Supported: {sorted(_ENGINES)}")
    module_path, _, class_name = registry[engine].rpartition(".")
    return getattr(importlib.import_module(module_path), class_name)


def make_calculator(
    mtp_data: MTPData,
    engine: str = "cext",
    *,
    relax_magmoms: bool | None = None,
    static_geometry: bool = False,
    **kwargs: dict,
) -> "MTP | MMTP":
    """Create the appropriate calculator based on data type.

    Parameters
    ----------
    mtp_data : MTPData
        Potential data.  If this is a ``MagMTPData`` the magnetic calculator
        is returned automatically.
    engine : str
        Backend name (``"cext"``, ``"numba"``, etc.).
    relax_magmoms : bool or None
        Whether to relax magnetic moments.  ``None`` (default) resolves to
        ``True`` for a dynamic geometry, ``False`` when ``static_geometry``.
    static_geometry : bool
        If True, the atomic geometry is assumed fixed across calls (training).
    **kwargs
        Forwarded to the calculator constructor (e.g. ``warm_start_magmoms``).

    Returns
    -------
    ``MMTP`` calculator if *mtp_data* is a ``MagMTPData``
    instance, otherwise a plain ``MTP`` calculator.

    """
    if isinstance(mtp_data, MagMTPData):
        if relax_magmoms is None:
            relax_magmoms = not static_geometry
        return MMTP(
            mtp_data,
            engine=engine,
            relax_magmoms=relax_magmoms,
            static_geometry=static_geometry,
            **kwargs,
        )
    return MTP(mtp_data, engine=engine, static_geometry=static_geometry, **kwargs)


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
        static_geometry: bool = False,
        **kwargs: dict,
    ) -> None:
        super().__init__(*args, **kwargs)
        magnetic = isinstance(mtp_data, MagMTPData)
        engine_cls = make_mtp_engine(engine, magnetic=magnetic)
        self.engine: EngineBase = engine_cls(mtp_data, static_geometry=static_geometry)

    def update_parameters(self, mtp_data: MTPData) -> None:
        """Update MTP parameters."""
        if self.engine.update(mtp_data):
            self.results = {}

    def _store(self, atoms: Atoms, results: dict) -> None:
        self.results = results
        self.results["free_energy"] = self.results["energy"]
        if atoms.cell.rank != 3 and "stress" in self.results:  # noqa: PLR2004
            del self.results["stress"]

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] = ["energy"],
        system_changes: list[str] = all_changes,
    ) -> None:
        """Calculate."""
        super().calculate(atoms, properties, system_changes)
        self._store(self.atoms, self.engine.efs(self.atoms))

    def compute_jacobian(self, atoms: Atoms) -> None:
        """Run a Jacobian pass, populating engine basis data and results."""
        self.atoms = atoms
        self._store(atoms, self.engine.jac(atoms))


class MMTP(MTP):
    """ASE Calculator of the magnetic MTP potential.

    Parameters
    ----------
    mtp_data : MTPData | MagMTPData
        Potential data.  If a plain ``MTPData`` is passed it is automatically
        promoted to ``MagMTPData`` via ``MagMTPData.from_base()``.
    engine : str
        Engine backend to use.
    static_geometry : bool
        If True, the atomic geometry is assumed fixed across calls (training).
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
        mtp_data: MTPData,
        *args,
        engine: str = "cext",
        static_geometry: bool = False,
        **kwargs,
    ) -> None:
        if not isinstance(mtp_data, MagMTPData):
            mtp_data = MagMTPData.from_base(mtp_data)
        Calculator.__init__(self, *args, **kwargs)
        engine_cls = make_mtp_engine(engine, magnetic=True)
        self.engine: MagEngineBase = engine_cls(
            mtp_data, static_geometry=static_geometry
        )
        self._relaxed_magmoms: np.ndarray | None = None

    def update_parameters(self, mtp_data: MagMTPData) -> None:
        if self.engine.update(mtp_data):
            self.results = {}

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

            results = self.engine.relax_magnetic_moments(
                self.atoms, magmoms_init=magmoms_init
            )
            self._relaxed_magmoms = results["magmoms"].copy()
            self._store(self.atoms, results)
        else:
            self._store(self.atoms, self.engine.efs(self.atoms))

    def compute_jacobian(self, atoms: Atoms, *, mgrad: bool = False) -> None:
        """Run a Jacobian pass, populating engine basis data and results."""
        self.atoms = atoms
        self._store(atoms, self.engine.jac(atoms, mgrad=mgrad))
