"""Loss function."""

import copy

import mlippy
from ase.data import chemical_symbols

from motep.io.mlip.mtp import read_mtp, write_mtp
from motep.loss_function import LossFunctionBase, calc_properties, update_mtp


def init_mlip(file: str, species: list[str]):
    mlip = mlippy.initialize()
    mlip = mlippy.mtp()
    mlip.load_potential(file)
    for _ in species:
        mlip.add_atomic_type(chemical_symbols.index(_))
    return mlip


def MTP_field(
    file: str,
    untrained_mtp: str,
    parameters: list[float],
    species: list[str],
):
    data = read_mtp(untrained_mtp)
    data = update_mtp(copy.deepcopy(data), parameters)

    write_mtp(file, data)
    mlip = init_mlip(file, species)
    potential = mlippy.MLIP_Calculator(mlip, {})

    return potential


class MlippyLossFunction(LossFunctionBase):
    def __call__(self, parameters: list[float]) -> float:
        file = self.setting["potential_final"]
        calc = MTP_field(file, self.untrained_mtp, parameters, self.species)
        for atoms in self.images:
            atoms.calc = calc
        energies, forces, stresses = calc_properties(self.images, self.comm)
        return self.calc_loss_function(energies, forces, stresses)

    def calc_rmses(self, parameters: list[float]) -> dict[str, float]:
        """Calculate RMSEs."""
        file = self.setting["potential_final"]
        calc = MTP_field(file, self.untrained_mtp, parameters, self.species)
        for atoms in self.images:
            atoms.calc = calc
        super().calc_rmses(parameters)
