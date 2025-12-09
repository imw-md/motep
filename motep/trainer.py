"""`motep train` command."""

import argparse
import logging
import pathlib
from pprint import pformat

import numpy as np
from ase import Atoms
from mpi4py import MPI

from motep.io.mlip.mtp import read_mtp, write_mtp
from motep.io.utils import get_dummy_species, read_images
from motep.loss import ErrorPrinter, LossFunction
from motep.optimizers import OptimizerBase, make_optimizer
from motep.potentials.mtp.data import MTPData
from motep.setting import TrainSetting, load_setting_train
from motep.utils import measure_time

logger = logging.getLogger(__name__)


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

    def update_mindist(self, images: list[Atoms]) -> None:
        """Update min_dist of the MTP potential."""
        self.mtp_data.min_dist = np.min([_.get_all_distances(mic=True) for _ in images])

    def train(self, images: list[Atoms]) -> LossFunction:
        """Train."""
        setting = self.setting

        if setting.update_mindist:
            self.update_mindist(images)

        loss_args = (images, self.mtp_data, setting.loss)
        if setting.engine == "mlippy":
            from motep.mlippy_loss import MlippyLossFunction

            loss = MlippyLossFunction(*loss_args, comm=self.comm)
        else:
            loss = LossFunction(*loss_args, engine=setting.engine, comm=self.comm)

        for i, step in enumerate(setting.steps):
            with measure_time(f"step {i}: {step['method']}", self.comm):
                if self.comm.rank == 0:
                    logger.info(f"{'':=^72s}\n")
                    logger.info(pformat(step))
                    logger.info("")
                    for handler in logger.handlers:
                        handler.flush()

                # Print parameters before optimization.
                self.mtp_data.initialize(setting.rng)
                if self.comm.rank == 0:
                    self.mtp_data.log()

                # Instantiate an `Optimizer` class
                optimizer: OptimizerBase = make_optimizer(step["method"])(loss, **step)
                optimizer.optimize(**step.get("kwargs", {}))
                loss.broadcast()  # be sure that all processes have the same data
                if self.comm.rank == 0:
                    logger.info("")
                    for handler in logger.handlers:
                        handler.flush()

                    # Print parameters after optimization.
                    self.mtp_data.log()

                    write_mtp(f"intermediate_{i}.mtp", self.mtp_data)

                    ErrorPrinter(loss).log()
        return loss


def train(filename_setting: str, comm: MPI.Comm) -> None:
    """Train."""
    setting = load_setting_train(filename_setting)
    if comm.rank == 0:
        logger.info(pformat(setting))
        logger.info("")
        for handler in logger.handlers:
            handler.flush()

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
    trainer.train(images)

    if comm.rank == 0:
        logger.info(f"{'':=^72s}\n")
        write_mtp(setting.potential_final, mtp_data)


def run(args: argparse.Namespace) -> None:
    """Run."""
    comm = MPI.COMM_WORLD
    with measure_time("total"):
        train(args.setting, comm)
