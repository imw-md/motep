"""Loss function."""

import pathlib
import tempfile

import mlippy

from motep.io.mlip.mtp import write_mtp
from motep.loss import LossFunctionBase


def init_mlip(file: str, species: list[int]):
    mlip = mlippy.initialize()
    mlip = mlippy.mtp()
    mlip.load_potential(file)
    for _ in species:
        mlip.add_atomic_type(_)
    return mlip


class MlippyLossFunction(LossFunctionBase):
    """Loss function with proper treatment for mlippy."""

    def __init__(self, *args: tuple, **kwargs: dict) -> None:
        super().__init__(*args, **kwargs)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as fd:
            file = fd.name
        write_mtp(file, self.mtp_data)
        mlip = init_mlip(file, self.mtp_data.species)
        for atoms in self.images:
            targets = atoms.calc.results
            atoms.calc = mlippy.MLIP_Calculator(mlip, {})
            atoms.calc.targets = targets
        pathlib.Path.unlink(file)

    def __call__(self, parameters: list[float]) -> float:
        self.mtp_data.parameters = parameters
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as fd:
            file = fd.name
        write_mtp(file, self.mtp_data)
        options = {"mtp-filename": file}
        for atoms in self.images:
            atoms.calc.mlip.init_wrapper(options)
            atoms.calc.results = {}
        pathlib.Path.unlink(file)
        return self.calc_loss_function()
