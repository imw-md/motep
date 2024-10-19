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


def init_calc(file: str, mtp_data: MTPData, species: list[int]):
    """Initialize the mlippy `mtp` ojbect."""
    write_mtp(file, mtp_data)
    return init_mlip(file, species)


class MlippyLossFunction(LossFunctionBase):
    def __init__(self, *args: tuple, **kwargs: dict) -> None:
        super().__init__(*args, **kwargs)
        file = self.setting["potential_initial"]
        mlip = init_calc(file, self.mtp_data, self.setting["species"])
        for atoms in self.images:
            targets = atoms.calc.results
            atoms.calc = mlippy.MLIP_Calculator(mlip, {})
            atoms.calc.targets = targets

    def __call__(self, parameters: list[float]) -> float:
        file = self.setting["potential_final"]
        self.mtp_data.parameters = parameters
        write_mtp(file, self.mtp_data)
        options = {"mtp-filename": file}
        for atoms in self.images:
            atoms.calc.mlip.init_wrapper(options)
            atoms.calc.results = {}
        return self.calc_loss_function()
