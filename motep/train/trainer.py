"""`motep train`."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms
from scipy.optimize._minimize import MINIMIZE_METHODS  # noqa: PLC2701

from motep.io.mlip.mtp import read_mtp, write_mtp
from motep.io.utils import get_dummy_species, read_images
from motep.loss import ErrorPrinter, LossFunction, LossFunctionBase, LossSetting
from motep.optimizers import make_optimizer
from motep.parallel import DummyMPIComm, world
from motep.potentials.mtp.data import MTPData
from motep.setting import DataclassFromAny, Setting, parse_setting
from motep.utils import measure_time

if TYPE_CHECKING:
    from motep.optimizers.base import OptimizerBase

logger = logging.getLogger(__name__)


def _convert_steps(steps: list[dict]) -> list[dict]:
    for i, value in enumerate(steps):
        if not isinstance(value, dict):
            steps["steps"][i] = {"method": value}
        if value["method"].lower() in MINIMIZE_METHODS:
            if "kwargs" not in value:
                value["kwargs"] = {}
            value["kwargs"]["method"] = value["method"]
            value["method"] = "minimize"
    return steps


@dataclass
class TrainPotentials(DataclassFromAny):
    """Setting of the potentials."""

    initial: str = "initial.mtp"
    final: str = "final.mtp"


@dataclass
class TrainSetting(Setting):
    """Setting of the training."""

    potentials: TrainPotentials = field(default_factory=TrainPotentials)
    loss: LossSetting = field(default_factory=LossSetting)
    steps: list[dict] = field(default_factory=lambda: [{"method": "minimize"}])
    update_mindist: bool = False

    def __post_init__(self) -> None:
        """Postprocess attributes."""
        if isinstance(self.loss, dict):
            self.loss = LossSetting(**self.loss)

        # Default 'optimized' is defined in each `Optimizer` class.

        # convert the old style "steps" like {'steps`: ['L-BFGS-B']} to the new one
        # {'steps`: {'method': 'L-BFGS-B'}
        self.steps = _convert_steps(self.steps)


def load_setting_train(filename: str | Path | None = None) -> TrainSetting:
    """Load setting for `train`.

    Returns
    -------
    TrainSetting

    """
    if filename is None:
        return TrainSetting()
    return TrainSetting(**parse_setting(filename))


class Trainer:
    """Trainer."""

    def __init__(
        self,
        mtp_data: MTPData,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
        engine: str = "cext",
        loss: dict | LossSetting | None = None,
        steps: list[dict] | None = None,
        *,
        update_mindist: bool = False,
        comm: DummyMPIComm = world,
    ) -> None:
        """Initialize.

        Parameters
        ----------
        mtp_data : MTPData
            MTPData object with MTP parameters.
        seed : int | None (optional)
            Seed for the random number generator. Disregarded if `rng` is given.
        rng : np.random.Generator | None (optional)
            Pseudo-random-number generator (PRNG) with the NumPy API.
        engine : str (optional)
            Engine name.
        loss : dict | LossSetting | None (optional)
            Dict with settings of the loss function.
        steps : list[dict] | None (optional)
            List of optimization steps.
        comm : MPI.Comm
            MPI.Comm object.
        update_mindist : bool (optional)
            Whether to update min_dist of the MTP potential before training.

        """
        self.mtp_data = mtp_data

        seed = seed or comm.bcast(np.random.SeedSequence().entropy % (2**32), root=0)
        if seed is not None and comm.rank == 0:
            logger.info("[random seed] = %d", seed)
        self.rng = rng or np.random.default_rng(seed)

        self.engine = engine
        self.loss = LossSetting.from_any(loss)
        self.steps = steps or [{"method": "minimize"}]
        self.comm = comm
        self.should_update_mindist = update_mindist

    def update_mindist(self, images: list[Atoms]) -> None:
        """Update min_dist of the MTP potential."""
        self.mtp_data.min_dist = np.min([_.get_all_distances(mic=True) for _ in images])

    def train(self, images: list[Atoms]) -> LossFunctionBase:
        """Train.

        Parameters
        ----------
        images : list[Atoms]
            List of ASE Atoms objects.

        Returns
        -------
        loss : LossFunctionBase
            LossFunction object after training.

        """
        if self.should_update_mindist:
            self.update_mindist(images)

        loss_args = (images, self.mtp_data, self.loss)
        if self.engine == "mlippy":
            from motep.mlippy_loss import MlippyLossFunction

            loss = MlippyLossFunction(*loss_args, comm=self.comm)
        else:
            loss = LossFunction(*loss_args, engine=self.engine, comm=self.comm)

        for i, step in enumerate(self.steps):
            with measure_time(f"step {i}: {step['method']}", comm=self.comm):
                if self.comm.rank == 0:
                    logger.info("%s\n", "=" * 72)
                    logger.info(pformat(step))
                    logger.info("")
                    for handler in logger.handlers:
                        handler.flush()

                # Print parameters before optimization.
                self.mtp_data.initialize(self.rng)
                if self.comm.rank == 0:
                    self.mtp_data.log()

                # Instantiate an `Optimizer` class
                optimizer: OptimizerBase = make_optimizer(step["method"])(loss, **step)
                optimizer.optimize(**step.get("kwargs", {}))
                loss.broadcast_results()
                if self.comm.rank == 0:
                    logger.info("")
                    for handler in logger.handlers:
                        handler.flush()

                    # Print parameters after optimization.
                    self.mtp_data.log()

                    write_mtp(f"intermediate_{i}.mtp", self.mtp_data)

                    ErrorPrinter(loss.images).log()
        return loss


def train_from_setting(filename_setting: str, comm: DummyMPIComm) -> None:
    """Train."""
    setting = load_setting_train(filename_setting)
    if comm.rank == 0:
        logger.info(pformat(setting))
        logger.info("")
        for handler in logger.handlers:
            handler.flush()

    untrained_mtp = str(Path(setting.potentials.initial).resolve())

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

    trainer = Trainer(
        mtp_data,
        seed=setting.seed,
        engine=setting.engine,
        loss=setting.loss,
        steps=setting.steps,
        comm=comm,
        update_mindist=setting.update_mindist,
    )
    trainer.train(images)

    if comm.rank == 0:
        logger.info("%s\n", "=" * 72)
        write_mtp(setting.potentials.final, mtp_data)
