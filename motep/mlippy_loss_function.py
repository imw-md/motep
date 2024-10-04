"""Loss function."""

import mlippy
from ase.data import chemical_symbols

from motep.initializer import MTPData
from motep.io.mlip.mtp import write_mtp
from motep.loss_function import LossFunctionBase, calc_properties


def init_mlip(file: str, species: list[str]):
    mlip = mlippy.initialize()
    mlip = mlippy.mtp()
    mlip.load_potential(file)
    for _ in species:
        mlip.add_atomic_type(chemical_symbols.index(_))
    return mlip


def init_calc(
    file: str,
    data: MTPData,
    parameters: list[float],
    species: list[str],
) -> mlippy.MLIP_Calculator:
    """Initialize mlippy ASE calculator."""
    data.update(parameters)
    write_mtp(file, data.data)
    mlip = init_mlip(file, species)
    return mlippy.MLIP_Calculator(mlip, {})


class MlippyLossFunction(LossFunctionBase):
    def __call__(self, parameters: list[float]) -> float:
        file = self.setting["potential_final"]
        calc = init_calc(file, self.data, parameters, self.species)
        for atoms in self.images:
            atoms.calc = calc
        energies, forces, stresses = calc_properties(self.images, self.comm)
        return self.calc_loss_function(energies, forces, stresses)

    def print_errors(self, parameters: list[float]) -> dict[str, float]:
        """Calculate RMSEs."""
        file = self.setting["potential_final"]
        calc = init_calc(file, self.data, parameters, self.species)
        for atoms in self.images:
            atoms.calc = calc
        super().print_errors(parameters)
