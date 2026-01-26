"""`motep run` command."""

import argparse
import logging
import pathlib
from pprint import pformat

from mpi4py import MPI

import motep.io
from motep.calculator import MTP
from motep.io.mlip.mtp import read_mtp
from motep.io.utils import get_dummy_species, read_images
from motep.setting import load_setting_apply

logger = logging.getLogger(__name__)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments."""
    parser.add_argument("setting")


def apply(filename_setting: str, comm: MPI.Comm) -> None:
    """Run."""
    setting = load_setting_apply(filename_setting)
    if comm.rank == 0:
        logger.info(pformat(setting))
        logger.info("")
        for handler in logger.handlers:
            handler.flush()

    mtp_file = str(pathlib.Path(setting.potential_final).resolve())

    species = setting.species or None
    images_in = read_images(
        setting.data_in,
        species=species,
        comm=comm,
        title="data_in",
    )
    if not setting.species:
        species = get_dummy_species(images_in)

    mtp_data = read_mtp(mtp_file)
    mtp_data.species = species

    for atoms in images_in:
        atoms.calc = MTP(mtp_data, engine=setting.engine, is_trained=False)
        atoms.get_potential_energy()

    motep.io.write(setting.data_out[0], images_in)


def run(args: argparse.Namespace) -> None:
    """Run."""
    comm = MPI.COMM_WORLD
    apply(args.setting, comm)
