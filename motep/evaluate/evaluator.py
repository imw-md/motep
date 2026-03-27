"""`motep evaluate` command."""

import logging
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat

from motep.calculator import MTP
from motep.io.mlip.mtp import read_mtp
from motep.io.utils import get_dummy_species, read_images
from motep.loss import ErrorPrinter
from motep.parallel import DummyMPIComm
from motep.potentials.mtp.data import MTPData
from motep.setting import Setting, parse_setting

logger = logging.getLogger(__name__)


@dataclass
class EvaluateSetting(Setting):
    """Setting for the application of the potential."""


def load_setting_evaluate(filename: str | Path | None = None) -> EvaluateSetting:
    """Load setting for `evaluate`.

    Returns
    -------
    EvaluateSetting

    """
    if filename is None:
        return EvaluateSetting()
    return EvaluateSetting(**parse_setting(filename))


class Evaluator:
    """Evaluator for MTP potential on test data."""

    def __init__(
        self,
        mtp_data: MTPData,
        engine: str = "cext",
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

        for i, atoms in enumerate(images_eval):
            # Save targets before replacing calculator
            targets = atoms.calc.results if atoms.calc else {}
            atoms.calc = MTP(self.mtp_data, engine=self.engine, mode="run")
            atoms.calc.targets = targets
            energy = atoms.get_potential_energy()
            logger.info("configuration %d: %s", i, energy)

        return images_eval


def evaluate_from_setting(filename_setting: str, comm: DummyMPIComm) -> None:
    """Evaluate the MTP potential on data from a setting file and print errors.

    Parameters
    ----------
    filename_setting : str
        Path to the setting file.
    comm : MPI.Comm
        MPI communicator.

    """
    setting = load_setting_evaluate(filename_setting)
    if comm.rank == 0:
        logger.info(pformat(setting))
        logger.info("")
        for handler in logger.handlers:
            handler.flush()

    mtp_file = str(Path(setting.potential_final).resolve())

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
