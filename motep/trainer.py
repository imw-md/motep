"""`motep train` command."""

import argparse
import logging
import pathlib
from pprint import pformat

import numpy as np
from mpi4py import MPI

from motep.io.mlip.mtp import read_mtp, write_mtp
from motep.io.utils import get_dummy_species, read_images
from motep.loss import ErrorPrinter, LossFunction
from motep.optimizers import OptimizerBase, make_optimizer
from motep.setting import load_setting_train
from motep.utils import measure_time

logger = logging.getLogger(__name__)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments."""
    parser.add_argument("setting")


def train(filename_setting: str, comm: MPI.Comm) -> None:
    """Train."""
    rank = comm.Get_rank()

    setting = load_setting_train(filename_setting)
    if rank == 0:
        logger.info(pformat(setting))
        logger.info("")
        for handler in logger.handlers:
            handler.flush()

    setting.rng = np.random.default_rng(setting.seed)

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

    if setting.engine == "mlippy":
        from motep.mlippy_loss import MlippyLossFunction

        loss = MlippyLossFunction(images, mtp_data, setting.loss, comm=comm)
    else:
        loss = LossFunction(
            images,
            mtp_data,
            setting.loss,
            engine=setting.engine,
            comm=comm,
        )

    for i, step in enumerate(setting.steps):
        with measure_time(f"step {i}: {step['method']}", comm):
            if rank == 0:
                logger.info(f"{'':=^72s}\n")
                logger.info(pformat(step))
                logger.info("")
                for handler in logger.handlers:
                    handler.flush()

            # Print parameters before optimization.
            mtp_data.initialize(setting.rng)
            if rank == 0:
                mtp_data.log()

            # Instantiate an `Optimizer` class
            optimizer: OptimizerBase = make_optimizer(step["method"])(loss, **step)
            optimizer.optimize(**step.get("kwargs", {}))
            loss.broadcast()  # be sure that all processes have the same data
            if rank == 0:
                logger.info("")
                for handler in logger.handlers:
                    handler.flush()

                # Print parameters after optimization.
                mtp_data.log()

                write_mtp(f"intermediate_{i}.mtp", mtp_data)

                ErrorPrinter(loss).log()

    if rank == 0:
        logger.info(f"{'':=^72s}\n")
        write_mtp(setting.potential_final, mtp_data)


def run(args: argparse.Namespace) -> None:
    """Run."""
    comm = MPI.COMM_WORLD
    with measure_time("total"):
        train(args.setting, comm)
