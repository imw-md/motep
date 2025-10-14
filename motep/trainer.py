"""`motep train` command."""

import argparse
import pathlib
import pprint

from ase import Atoms
from mpi4py import MPI

from motep.io.mlip.mtp import read_mtp, write_mtp
from motep.io.utils import get_dummy_species, read_images
from motep.loss import ErrorPrinter, LossFunction
from motep.optimizers import OptimizerBase, make_optimizer
from motep.potentials.mtp.data import MTPData
from motep.setting import TrainSetting, load_setting_train
from motep.utils import measure_time


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments."""
    parser.add_argument("setting")


class Trainer:
    """Trainer."""

    def __init__(
        self,
        mtp_data: MTPData,
        setting: TrainSetting,
        comm: MPI.Comm,
    ) -> None:
        """Initialize."""
        self.mtp_data = mtp_data
        self.setting = setting
        self.comm = comm

    def train(self, images: list[Atoms]) -> LossFunction:
        """Train."""
        loss_args = (images, self.mtp_data, self.setting.loss)
        loss = LossFunction(*loss_args, engine=self.setting.engine, comm=self.comm)

        for step in self.setting.steps:
            self.mtp_data.initialize(self.setting.rng)
            optimizer: OptimizerBase = make_optimizer(step["method"])(loss, **step)
            optimizer.optimize(**step.get("kwargs", {}))
            loss.broadcast()

        return loss

    def train_verbose(self, images: list[Atoms]) -> LossFunction:
        setting = self.setting
        loss_args = (images, self.mtp_data, setting.loss)
        if self.setting.engine == "mlippy":
            from motep.mlippy_loss import MlippyLossFunction

            loss = MlippyLossFunction(*loss_args, comm=self.comm)
        else:
            loss = LossFunction(*loss_args, engine=setting.engine, comm=self.comm)

        for i, step in enumerate(setting.steps):
            with measure_time(f"step {i}: {step['method']}", self.comm):
                if self.comm.rank == 0:
                    print(f"{'':=^72s}\n")
                    pprint.pp(step)
                    print(flush=True)

                # Print parameters before optimization.
                self.mtp_data.initialize(setting.rng)
                if self.comm.rank == 0:
                    self.mtp_data.print(flush=True)

                # Instantiate an `Optimizer` class
                optimizer: OptimizerBase = make_optimizer(step["method"])(loss, **step)
                optimizer.optimize(**step.get("kwargs", {}))
                loss.broadcast()  # be sure that all processes have the same data
                if self.comm.rank == 0:
                    print(flush=True)

                    # Print parameters after optimization.
                    self.mtp_data.print(flush=True)

                    write_mtp(f"intermediate_{i}.mtp", self.mtp_data)

                    ErrorPrinter(loss).print(flush=True)


def train(filename_setting: str, comm: MPI.Comm) -> None:
    """Train."""
    setting = load_setting_train(filename_setting)
    if comm.rank == 0:
        pprint.pp(setting)
        print(flush=True)

    untrained_mtp = str(pathlib.Path(setting.potential_initial).resolve())

    species = setting.species or None
    images = read_images(
        setting.data_training,
        species=species,
        comm=comm,
        title="data_training",
    )
    if not setting.species:
        species = get_dummy_species(images)

    mtp_data = read_mtp(untrained_mtp)
    mtp_data.species = species

    trainer = Trainer(mtp_data, setting, comm)
    trainer.train_verbose(images)

    if comm.rank == 0:
        print(f"{'':=^72s}\n")
        write_mtp(setting.potential_final, mtp_data)


def run(args: argparse.Namespace) -> None:
    """Run."""
    comm = MPI.COMM_WORLD
    with measure_time("total"):
        train(args.setting, comm)
