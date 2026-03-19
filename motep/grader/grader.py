"""Grader."""

import logging
import pathlib
from dataclasses import dataclass, field
from pprint import pformat

import numpy as np
from ase import Atoms

import motep.io
from motep.calculator import MTP
from motep.grader.maxvol import MaxVol, MaxVolSetting
from motep.io.mlip.mtp import read_mtp
from motep.io.utils import get_dummy_species, read_images
from motep.parallel import DummyMPIComm, world
from motep.potentials.mtp.data import MTPData
from motep.setting import Setting, parse_setting

logger = logging.getLogger(__name__)


@dataclass
class GradeSetting(Setting):
    """Setting for the extrapolation-grade calculations."""

    mode: str = "configuration"
    maxvol: MaxVolSetting = field(default_factory=MaxVolSetting)

    def __post_init__(self) -> None:
        """Postprocess."""
        self.maxvol = MaxVolSetting.from_any(self.maxvol)


def load_setting_grade(filename: str) -> GradeSetting:
    """Load setting for `grade`.

    Returns
    -------
    GradeSetting

    """
    return GradeSetting(**parse_setting(filename))


class Grader:
    """Grader."""

    def __init__(
        self,
        mtp_data: MTPData,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
        engine: str = "cext",
        *,
        mode: str = "configuration",
        maxvol_setting: MaxVolSetting | dict | None = None,
        comm: DummyMPIComm = world,
    ) -> None:
        """Initialize."""
        maxvol_setting = MaxVolSetting.from_any(maxvol_setting)

        self.mtp_data = mtp_data

        seed = seed or comm.bcast(np.random.SeedSequence().entropy % (2**32), root=0)
        if seed is not None and comm.rank == 0:
            logger.info("[random seed] = %d", seed)
        self.rng = rng or np.random.default_rng(seed)

        self.engine = engine
        self.maxvol_setting = maxvol_setting
        self.mode = mode
        self.comm = comm
        self.indices = np.array(np.iinfo(np.int32).min, dtype=int)
        self.active_set_matrix = np.array(np.nan)

    def update(self, images: list[Atoms]) -> None:
        """Reevaluate the matrix and active set.

        Parameters
        ----------
        images : list[Atoms]
            List of ASE Atoms objects used for training.

        """
        matrix = self.calc_matrix(images)
        maxvol = MaxVol(
            algorithm=self.maxvol_setting.algorithm,
            init_method=self.maxvol_setting.init_method,
            rng=self.rng,
        )
        self.indices = maxvol.run(
            matrix,
            threshold=self.maxvol_setting.threshold,
            maxiter=self.maxvol_setting.maxiter,
        )
        self.active_set_matrix = matrix[self.indices]

    def calc_matrix(self, images: list[Atoms]) -> np.ndarray:
        """Calculate the matrix of moment basis values.

        Returns
        -------
        matrix : np.ndarray

        Raises
        ------
        ValueError

        """
        for atoms in images:
            atoms.calc = MTP(self.mtp_data, engine=self.engine, mode="run")
            atoms.get_potential_energy()

        # Make the overdetermined matrix
        if self.mode == "configuration":
            return np.array([atoms.calc.engine.mbd.values for atoms in images])
        raise ValueError(self.mode)

    def grade(self, images: list[Atoms]) -> None:
        """Grade.

        Parameters
        ----------
        images : list[Atoms]
            List of ASE Atoms objects to evaluate.

        Raises
        ------
        ValueError

        """
        # Make the overdetermined matrix
        matrix = self.calc_matrix(images)
        # Eq. (8) or the one after Eq. (11) in [Podryabinkin_CMS_2017_Active]_
        c = np.linalg.lstsq(self.active_set_matrix.T, matrix.T, rcond=None)[0].T
        grades = np.max(c, axis=1)

        if self.mode == "configuration":
            # evaluate `MV_grade` for each configuration
            for atoms, maxvol_grade in zip(images, grades, strict=True):
                atoms.info["MV_grade"] = maxvol_grade
        else:
            raise ValueError(self.mode)


def grade(filename_setting: str, comm: DummyMPIComm = world) -> None:
    """Grade.

    This adds `MV_grade` to `atoms.info` or `nbh_grades` to `atoms.arrays`.

    Raises
    ------
    ValueError
        If `engine` does not support the extrapolation grades.

    """
    setting = load_setting_grade(filename_setting)
    if comm.rank == 0:
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
    images_in = read_images(
        setting.data_in,
        species=species,
        comm=comm,
        title="data_in",
    )

    mtp_data = read_mtp(mtp_file)
    mtp_data.species = species

    if setting.engine == "mlippy":
        msg = "`mlippy` engine is not available for `motep grade`"
        raise ValueError(msg)

    grader = Grader(
        mtp_data,
        engine=setting.engine,
        rng=rng,
        mode=setting.mode,
        maxvol_setting=MaxVolSetting.from_any(setting.maxvol),
        comm=comm,
    )
    grader.update(images_training)
    grader.grade(images_in)

    if comm.rank == 0:
        logger.info(f"{'':=^72s}\n")
        logger.info("[data_active]")
        logger.info(grader.indices)
        for handler in logger.handlers:
            handler.flush()
        motep.io.write(setting.data_out[0], images_in)
