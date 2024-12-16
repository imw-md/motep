"""ASE Calculators."""

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from motep.potentials.mtp.base import EngineBase
from motep.potentials.mtp.data import MTPData


def make_mtp_engine(engine: str = "numpy") -> EngineBase:
    if engine == "numpy":
        from motep.potentials.mtp.numpy.engine import NumpyMTPEngine

        return NumpyMTPEngine
    elif engine == "numba":
        from motep.potentials.mtp.numba.engine import NumbaMTPEngine

        return NumbaMTPEngine
    elif engine == "jax":
        from motep.potentials.mtp.jax.engine import JaxMTPEngine

        return JaxMTPEngine
    else:
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
        *args,
        engine: str = "numpy",
        is_trained: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.engine = make_mtp_engine(engine)(mtp_data, is_trained=is_trained)
        self.engine.update(mtp_data)

    def update_parameters(self, mtp_data: MTPData) -> None:
        self.engine.update(mtp_data)
        self.results = {}  # trigger new calculation

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] = ["energy"],
        system_changes: list[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)

        self.results = self.engine.calculate(self.atoms)

        self.results["free_energy"] = self.results["energy"]

        if self.atoms.cell.rank != 3 and "stress" in self.results:
            del self.results["stress"]
