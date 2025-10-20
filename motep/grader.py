"""`motep grade` command."""

import argparse
import logging
import pathlib
from pprint import pformat

import numpy as np
from mpi4py import MPI

import motep.io
from motep.active import AlgorithmBase, make_algorithm
from motep.io.mlip.mtp import read_mtp
from motep.io.utils import get_dummy_species, read_images
from motep.setting import load_setting_grade

logger = logging.getLogger(__name__)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments."""
    parser.add_argument("setting")


def grade(filename_setting: str, comm: MPI.Comm) -> None:
    """Grade.

    This adds `MV_grade` to `atoms.info`.
    """
    rank = comm.Get_rank()
    setting = load_setting_grade(filename_setting)
    if rank == 0:
        logger.info(pformat(setting))
        logger.info("")
        for handler in logger.handlers:
            handler.flush()

    rng = np.random.default_rng(setting.seed)

    mtp_file = str(pathlib.Path(setting.potential_final).resolve())

    species = setting.species or None
    images_training = read_images(
        setting.data_training,
        species=species,
        comm=comm,
        title="data_training",
    )
    if not setting.species:
        species = get_dummy_species(images_training)

    mtp_data = read_mtp(mtp_file)
    mtp_data.species = species

    if setting.engine == "mlippy":
        msg = "`mlippy` engine is not available for `motep grade`"
        raise ValueError(msg)

    algorithm_class = make_algorithm(setting.algorithm)

    optimality: AlgorithmBase = algorithm_class(
        images_training,
        mtp_data,
        setting.engine,
        rng=rng,
    )

    if rank == 0:
        logger.info(f"{'':=^72s}\n")
        logger.info("[data_active]")
        logger.info(optimality.indices)
        for handler in logger.handlers:
            handler.flush()

    images_in = read_images(
        setting.data_in,
        species=species,
        comm=comm,
        title="data_in",
    )

    optimality.calc_grade(images_in)

    motep.io.write(setting.data_out[0], images_in)


def run(args: argparse.Namespace) -> None:
    """Run."""
    comm = MPI.COMM_WORLD
    grade(args.setting, comm)
