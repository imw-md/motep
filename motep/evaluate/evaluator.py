"""`motep evaluate` command."""

import logging
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat

import motep.io
from motep.calculator import MTP
from motep.io.mlip.mtp import read_mtp
from motep.io.utils import get_dummy_species, read_images
from motep.loss import ErrorPrinter
from motep.parallel import DummyMPIComm
from motep.potentials.mtp.data import MTPData
from motep.setting import DataclassFromAny, Setting, parse_setting

logger = logging.getLogger(__name__)


@dataclass
class EvalConfigurations(DataclassFromAny):
    """Configurations."""

    initial: list[str] = field(default_factory=lambda: ["initial.cfg"])
    final: list[str] = field(default_factory=lambda: ["final.cfg"])


@dataclass
class EvalPotentials(DataclassFromAny):
    """Potentials."""

    final: str = "final.mtp"


@dataclass
class EvalSetting(Setting):
    """Setting for the application of the potential."""

    configurations: EvalConfigurations = field(default_factory=EvalConfigurations)
    potentials: EvalPotentials = field(default_factory=EvalPotentials)

    def __post_init__(self) -> None:
        """Postprocess attributes."""
        self.configurations = EvalConfigurations.from_any(self.configurations)
        self.potentials = EvalPotentials.from_any(self.potentials)


def load_setting_evaluate(filename: str | Path | None = None) -> EvalSetting:
    """Load setting for `evaluate`.

    Returns
    -------
    EvaluateSetting

    """
    if filename is None:
        return EvalSetting()
    return EvalSetting(**parse_setting(filename))


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

    mtp_file = str(Path(setting.potentials.final).resolve())

    species = setting.species or None
    images_initial = read_images(
        setting.configurations.initial,
        species=species,
        comm=comm,
        title="configurations.initial",
    )
    if not setting.species:
        species = get_dummy_species(images_initial)

    mtp_data = read_mtp(mtp_file)
    mtp_data.species = species

    # Run evaluation
    evaluator = Evaluator(mtp_data, engine=setting.engine)
    images_final = evaluator.evaluate(images_initial)

    # Print errors
    if comm.rank == 0:
        logger.info("%s\n", "=" * 72)
        ErrorPrinter(images_final).log()
        motep.io.write(setting.configurations.final[0], images_final)
