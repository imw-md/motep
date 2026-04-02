"""ASE Calculators."""

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

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
