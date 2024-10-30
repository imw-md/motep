"""ASE Calculators."""

from ase import Atoms
from ase.calculators.calculator import (
    Calculator,
    PropertyNotImplementedError,
    all_changes,
)

from motep.potentials.mtp.data import MTPData


def make_mtp_engine(engine: str = "numpy") -> type:
    if engine == "numpy":
        from motep.potentials.mtp.numpy.engine import NumpyMTPEngine

        return NumpyMTPEngine
    elif engine == "numba":
        from motep.potentials.mtp.numba.engine import NumbaMTPEngine

        return NumbaMTPEngine
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
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.engine = make_mtp_engine(engine)(mtp_data)
        self.engine.update(mtp_data)

    def update_parameters(self, dict_mtp: MTPData) -> None:
        self.engine.update(dict_mtp)
        self.results = {}  # trigger new calculation

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] = ["energy"],
        system_changes: list[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)

        energy, forces, stress = self.engine.calculate(self.atoms)

        self.results["energy"] = self.results["free_energy"] = energy
        self.results["forces"] = forces
        if self.atoms.cell.rank == 3:
            self.results["stress"] = stress
        elif "stress" in properties:
            raise PropertyNotImplementedError
