"""ASE Calculators."""

from typing import Any

from ase import Atoms
from ase.calculators.calculator import (
    Calculator,
    PropertyNotImplementedError,
    all_changes,
)


def make_mtp_engine(engine: str = "numpy"):
    if engine == "numpy":
        from .mtp import NumpyMTPEngine

        return NumpyMTPEngine()
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
        mtp_parameters: dict[str, Any],
        *args,
        engine: str = "numpy",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.engine = make_mtp_engine(engine)
        self.engine.update(mtp_parameters)

    def update_parameters(self, parameters: dict[str, Any]):
        self.engine.update(parameters)
        self.results = {}  # trigger new calculation

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] = ["energy"],
        system_changes: list[str] = all_changes,
    ):
        super().calculate(atoms, properties, system_changes)

        energy, forces, stress = self.engine.get_energy(self.atoms)

        self.results["energy"] = self.results["free_energy"] = energy
        self.results["forces"] = forces
        if self.atoms.cell.rank == 3:
            self.results["stress"] = stress
        elif "stress" in properties:
            raise PropertyNotImplementedError
