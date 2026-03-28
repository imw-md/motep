"""Grader."""

import logging
from copy import copy
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from pprint import pformat

import numpy as np
from ase import Atoms

import motep.io
from motep.calculator import MTP
from motep.grade.maxvol import MaxVol, MaxVolSetting
from motep.io.mlip.mtp import read_mtp
from motep.io.utils import get_dummy_species, read_images
from motep.parallel import DummyMPIComm, world
from motep.potentials.mtp.data import MTPData
from motep.setting import CommonSetting, DataclassFromAny, parse_setting

logger = logging.getLogger(__name__)


class GradeMode(StrEnum):
    """Extrapolation grade mode."""

    CONFIGURATION = "configuration"
    NEIGHBORHOOD = "neighborhood"


@dataclass
class GradeConfigurations(DataclassFromAny):
    """Configurations."""

    training: list[str] = field(default_factory=lambda: ["training.cfg"])
    initial: list[str] = field(default_factory=lambda: ["initial.cfg"])
    final: list[str] = field(default_factory=lambda: ["final.cfg"])


@dataclass
class GradePotentials(DataclassFromAny):
    """Setting of the potentials."""

    final: str = "final.mtp"


@dataclass
class GradeSetting(DataclassFromAny):
    """Setting for the extrapolation-grade calculations."""

    common: CommonSetting = field(default_factory=CommonSetting)
    configurations: GradeConfigurations = field(default_factory=GradeConfigurations)
    potentials: GradePotentials = field(default_factory=GradePotentials)
    mode: GradeMode = GradeMode.CONFIGURATION
    maxvol: MaxVolSetting = field(default_factory=MaxVolSetting)

    def __post_init__(self) -> None:
        """Postprocess attributes."""
        self.configurations = GradeConfigurations.from_any(self.configurations)
        self.potentials = GradePotentials.from_any(self.potentials)
        self.maxvol = MaxVolSetting.from_any(self.maxvol)


def load_setting_grade(filename: str | Path | None = None) -> GradeSetting:
    """Load setting for `grade`.

    Returns
    -------
    GradeSetting

    """
    if filename is None:
        return GradeSetting()
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
        mode: GradeMode = GradeMode.CONFIGURATION,
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
        matrix = self._calc_moment_basis_matrix(images)
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

    def _calc_moment_basis_matrix(self, images: list[Atoms]) -> np.ndarray:
        """Calculate the matrix of moment basis values.

        Returns
        -------
        moment_basis_matrix : np.ndarray

        Raises
        ------
        ValueError

        """
        # shallow copies of images
        images = [copy(_) for _ in images]

        for atoms in images:
            atoms.calc = MTP(self.mtp_data, engine=self.engine, mode="run")
            atoms.get_potential_energy()

        # Make the overdetermined matrix
        if self.mode == GradeMode.CONFIGURATION:
            return np.array([atoms.calc.engine.mbd.values for atoms in images])
        if self.mode == GradeMode.NEIGHBORHOOD:
            return np.vstack([atoms.calc.engine.mbd.vatoms.T for atoms in images])
        raise ValueError(self.mode)

    def grade(self, images: list[Atoms]) -> list[Atoms]:
        """Grade.

        Parameters
        ----------
        images : list[Atoms]
            List of ASE :class:`~ase.Atoms` objects to evaluate.

        Returns
        -------
        images : list[Atoms]
            List of ASE :class:`~ase.Atoms` objects with extrapolation grades.

        Raises
        ------
        ValueError

        Notes
        -----
        This class creates a lightweight shallow copy of the provided Atoms
        objects. Atomic positions and arrays are treated as immutable and are
        shared with the input. Only the calculator is replaced internally.

        """
        # shallow copies of images
        images = [copy(_) for _ in images]
        for atoms in images:
            atoms.calc = MTP(self.mtp_data, engine=self.engine, mode="run")
            atoms.get_potential_energy()

        # Make the overdetermined matrix
        matrix = self._calc_moment_basis_matrix(images)
        # Eq. (8) or the one after Eq. (11) in [Podryabinkin_CMS_2017_Active]_
        c = np.linalg.lstsq(self.active_set_matrix.T, matrix.T, rcond=None)[0].T
        grades = np.max(c, axis=1)

        if self.mode == GradeMode.CONFIGURATION:
            # evaluate `MV_grade` for each configuration
            for i, (atoms, maxvol_grade) in enumerate(zip(images, grades, strict=True)):
                atoms.calc.results["MV_grade"] = maxvol_grade
                logger.info("configuration %d: %s", i, maxvol_grade)
            return images
        if self.mode == GradeMode.NEIGHBORHOOD:
            idx = 0
            for i, atoms in enumerate(images):
                grades_per_image = grades[idx : idx + len(atoms)]
                maxvol_grade = grades_per_image.max()
                atoms.calc.results["nbh_grades"] = grades_per_image
                atoms.calc.results["MV_grade"] = maxvol_grade
                logger.info("configuration %d: %s", i, maxvol_grade)
                idx += len(atoms)
            return images
        raise ValueError(self.mode)


def grade_from_setting(filename_setting: str, comm: DummyMPIComm) -> None:
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

    rng = np.random.default_rng(setting.common.seed)

    mtp_file = str(Path(setting.potentials.final).resolve())

    species = setting.common.species or None
    images_training = read_images(
        setting.configurations.training,
        species=species,
        comm=comm,
        title="configurations.training",
    )
    if not setting.common.species:
        species = get_dummy_species(images_training)
    images_in = read_images(
        setting.configurations.initial,
        species=species,
        comm=comm,
        title="configurations.initial",
    )

    mtp_data = read_mtp(mtp_file)
    mtp_data.species = species

    if setting.common.engine == "mlippy":
        msg = "`mlippy` engine is not available for `motep grade`"
        raise ValueError(msg)

    grader = Grader(
        mtp_data,
        engine=setting.common.engine,
        rng=rng,
        mode=setting.mode,
        maxvol_setting=MaxVolSetting.from_any(setting.maxvol),
        comm=comm,
    )
    grader.update(images_training)
    images_out = grader.grade(images_in)

    if comm.rank == 0:
        logger.info("%s\n", "=" * 72)
        logger.info("[data_active]")
        logger.info(grader.indices)
        for handler in logger.handlers:
            handler.flush()
        motep.io.write(setting.configurations.final[0], images_out)
