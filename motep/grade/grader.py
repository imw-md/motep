"""Grader."""

import logging
from copy import copy
from pathlib import Path
from pprint import pformat

import numpy as np
from ase import Atoms

import motep.io
from motep.calculator import make_calculator
from motep.grade.maxvol import MaxVol, MaxVolResult, MaxVolSetting
from motep.io.mlip.mtp import read_mtp
from motep.io.utils import get_dummy_species, read_images
from motep.parallel import DummyMPIComm, world
from motep.potentials.mtp.data import MTPData

from .setting import GradeMode, load_setting_grade

logger = logging.getLogger(__name__)


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
        self.maxvol = MaxVol(
            algorithm=maxvol_setting.algorithm,
            init_method=maxvol_setting.init_method,
            threshold=maxvol_setting.threshold,
            maxiter=maxvol_setting.maxiter,
            rng=self.rng,
        )
        self.maxvol_result = MaxVolResult()
        self.mode = mode
        self.comm = comm

    def update(self, images: list[Atoms]) -> None:
        """Reevaluate the matrix and active set.

        Parameters
        ----------
        images : list[Atoms]
            List of ASE Atoms objects used for training.

        """
        matrix = self._calc_moment_basis_matrix(images)
        self.maxvol_result = self.maxvol.run(matrix)

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
            atoms.calc = make_calculator(
                self.mtp_data,
                engine=self.engine,
                mode="run",
                relax_magmoms=False,
            )
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
            atoms.calc = make_calculator(
                self.mtp_data,
                engine=self.engine,
                mode="run",
                relax_magmoms=False,
            )
            atoms.get_potential_energy()

        # Make the overdetermined matrix
        matrix = self._calc_moment_basis_matrix(images)

        active_set_matrix = self.maxvol_result.submatrix

        # Eq. (8) or the one after Eq. (11) in [Podryabinkin_CMS_2017_Active]_
        c = np.linalg.lstsq(active_set_matrix.T, matrix.T, rcond=None)[0].T
        grades = np.max(np.abs(c), axis=1)

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

    mtp_data = read_mtp(mtp_file)
    mtp_data.species = species

    if setting.common.engine == "mlippy":
        msg = "`mlippy` engine is not available for `motep grade`"
        raise ValueError(msg)

    grader = Grader(
        mtp_data,
        engine=setting.common.engine,
        rng=rng,
        mode=setting.grade.mode,
        maxvol_setting=MaxVolSetting.from_any(setting.grade.maxvol),
        comm=comm,
    )
    grader.update(images_training)

    initial = setting.configurations.initial
    final = setting.configurations.final
    for i, (filename_in, filename_out) in enumerate(zip(initial, final, strict=True)):
        images_in = read_images(
            [filename_in],
            species=species,
            comm=comm,
            title=f"configurations.initial[{i}]",
        )
        images_out = grader.grade(images_in)
        if comm.rank == 0:
            logger.info("%s\n", "=" * 72)
            logger.info("[data_active]")
            logger.info(grader.maxvol_result.indices)
            for handler in logger.handlers:
                handler.flush()
            motep.io.write(filename_out, images_out)
