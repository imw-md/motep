"""Loss function."""

import mlippy

from motep.io.mlip.mtp import write_mtp
from motep.loss_function import LossFunctionBase
from motep.potentials.mtp.data import MTPData


def init_mlip(file: str, species: list[int]):
    mlip = mlippy.initialize()
    mlip = mlippy.mtp()
    mlip.load_potential(file)
    for _ in species:
        mlip.add_atomic_type(_)
    return mlip


def init_calc(
    file: str,
    mtp_data: MTPData,
    parameters: list[float],
    species: list[int],
) -> mlippy.MLIP_Calculator:
    """Initialize mlippy ASE calculator."""
    mtp_data.parameters = parameters
    write_mtp(file, mtp_data)
    mlip = init_mlip(file, species)
    return mlippy.MLIP_Calculator(mlip, {})


class MlippyLossFunction(LossFunctionBase):
    def __call__(self, parameters: list[float]) -> float:
        file = self.setting["potential_final"]
        calc = init_calc(file, self.mtp_data, parameters, self.setting["species"])
        for atoms in self.images:
            atoms.calc = calc
        return self.calc_loss_function()
