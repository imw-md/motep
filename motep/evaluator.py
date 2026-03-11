"""`motep evaluate` command."""

import argparse
import logging
import pathlib
from copy import copy
from pprint import pformat

from mpi4py import MPI

from motep.calculator import MTP
from motep.io.mlip.mtp import read_mtp
from motep.io.utils import get_dummy_species, read_images
from motep.loss import ErrorPrinter
from motep.potentials.mtp.data import MTPData
from motep.setting import load_setting_apply
from motep.utils import measure_time

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator for MTP potential on test data."""

    def __init__(
        self,
        mtp_data: MTPData,
        engine: str = "numba",
    ) -> None:
        """Initialize Evaluator.

        Parameters
        ----------
        mtp_data : MTPData
            MTP potential data.
        engine : str
            Engine to use for calculations ("numpy", "numba", "jax", "cext", etc.).

        """
        self.mtp_data = mtp_data
        self.engine = engine

    def evaluate(self, images: list) -> list:
        """Run MTP calculations on images.

        Parameters
        ----------
        images : list[Atoms]
            List of ASE Atoms objects with targets stored in `atoms.calc.targets`.

        Returns
        -------
        list[Atoms]
            Images with computed results from MTP potential.

        """
        # Create shallow copies to preserve originals
        images_eval = [copy(_) for _ in images]

        for atoms in images_eval:
            # Save targets before replacing calculator
            targets = atoms.calc.results
            atoms.calc = MTP(self.mtp_data, engine=self.engine, mode="run")
            atoms.calc.targets = targets
            atoms.get_potential_energy()

        return images_eval


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments."""
    parser.add_argument("setting")


def evaluate_from_setting(filename_setting: str, comm: MPI.Comm) -> None:
    """Evaluate the MTP potential on data from a setting file and print errors.

    Parameters
    ----------
    filename_setting : str
        Path to the setting file.
    comm : MPI.Comm
        MPI communicator.

    """
    setting = load_setting_apply(filename_setting)
    if comm.rank == 0:
        logger.info(pformat(setting))
        logger.info("")
        for handler in logger.handlers:
            handler.flush()

    mtp_file = str(pathlib.Path(setting.potential_final).resolve())

    species = setting.species or None
    images = read_images(
        setting.data_in,
        species=species,
        comm=comm,
        title="data_in",
    )
    if not setting.species:
        species = get_dummy_species(images)

    mtp_data = read_mtp(mtp_file)
    mtp_data.species = species

    # Run evaluation
    evaluator = Evaluator(mtp_data, engine=setting.engine)
    images_eval = evaluator.evaluate(images)

    # Print errors
    if comm.rank == 0:
        logger.info(f"{'':=^72s}\n")
        ErrorPrinter(images_eval).log()


def run(args: argparse.Namespace) -> None:
    """Run."""
    comm = MPI.COMM_WORLD
    with measure_time("total", comm):
        evaluate_from_setting(args.setting, comm)
