"""Module for algorithms."""

import logging
from abc import ABC, abstractmethod
from itertools import combinations
from math import comb

import numpy as np
from ase import Atoms

from motep.calculator import MTP
from motep.potentials.mtp.data import MTPData

logger = logging.getLogger(__name__)


class AlgorithmBase(ABC):
    """Active-set finder.

    Attributes
    ----------
    indices : list[int]
        Indices for `loss.images` that consist of the active set.

    """

    def __init__(
        self,
        images_training: list[Atoms],
        mtp_data: MTPData,
        engine: str,
        rng: np.random.Generator,
        maxiter: int = 100_000,
    ) -> None:
        """Active-set finder."""
        self.images_training = images_training
        self.mtp_data = mtp_data
        self.engine = engine
        self.rng = rng
        self.maxiter = maxiter
        self.update()

    def update(self) -> None:
        """Reevaluates the matrix and active set."""
        self.matrix = self.calc_matrix(self.images_training)
        self.indices = []

        self.find_active_set()

    def calc_matrix(self, images: list[Atoms]) -> np.ndarray:
        """Calculate matrix.

        Parameters
        ----------
        images : list[Atoms]
            List of ASE :class:`~ase.Atoms` object.

        Returns
        -------
        np.ndarray
            Matrix to compute the extrapolation grade.

        """
        # Calculate basis functions of `images_training`
        for atoms in images:
            atoms.calc = MTP(self.mtp_data, engine=self.engine, mode="train")
            atoms.get_potential_energy()

        # Make the overdetermined matrix
        return np.array([atoms.calc.engine.mbd.values for atoms in images])

    @abstractmethod
    def find_active_set(self) -> None:
        """Find the active set."""

    def calc_grade(self, images: list[Atoms]) -> None:
        """Calculate the extrapolation grades."""
        # Make the overdetermined matrix
        matrix = self.calc_matrix(images)

        # Eq. (8) in [Podryabinkin_CMS_2017_Active]_
        matrix_active = self.matrix[self.indices]
        c = np.linalg.lstsq(matrix_active.T, matrix.T)[0].T
        maxvol_grades = np.max(c, axis=1)

        # evaluate `MV_grade` for each `atoms`
        for atoms, maxvol_grade in zip(images, maxvol_grades, strict=True):
            atoms.info["MV_grade"] = maxvol_grade


class ExhaustiveAlgorithm(AlgorithmBase):
    """Exhaustive algorithm."""

    def find_active_set(self) -> None:
        """Find the active set.

        Raises
        ------
        RuntimeError

        """
        images = self.images_training
        matrix = self.matrix

        # Choose rows (configurations)
        # This is preliminarily implemented only in an exhausive manner.
        # This is therefore valid so far only for a small `data_in`
        # and for a low level `potential_final`.
        asm = self.mtp_data.alpha_scalar_moments
        if comb(len(images), asm) > 2**24:  # 16777216
            msg = "too large possible combinations of rows"
            raise RuntimeError(msg, comb(len(images), asm))
        det_max = 0.0
        indices = np.arange(len(images))
        for _ in combinations(range(len(images)), asm):
            indices_checked = np.array(_)
            submatrix = matrix[indices_checked]
            det = np.abs(np.linalg.det(submatrix))
            if det > det_max:
                indices = indices_checked
                det_max = det

        self.indices = indices


class MaxVolAlgorithm(ExhaustiveAlgorithm):
    """MaxVol algorithm."""

    def find_active_set(self) -> None:
        """Find the active set.

        Raises
        ------
        RuntimeError

        """
        images = self.images_training
        matrix = self.matrix

        asm = self.mtp_data.alpha_scalar_moments
        indices = self.rng.choice(len(images), asm, replace=False)
        flags = np.zeros(len(images), dtype=bool)
        flags[indices] = True

        tolerance = 1e-9
        for _ in range(self.maxiter):
            submatrix = matrix[flags]
            c = np.linalg.lstsq(submatrix.T, matrix.T)[0].T
            i, j = np.divmod(np.argmax(np.abs(c)), asm)
            if np.abs(c[i, j]) < 1.0 + tolerance:
                break
            k = np.where(flags)[0][j]  # row/column in the original matrix
            flags[[k, i]] = flags[[i, k]]
        else:
            cmax = np.abs(c[i, j])
            msg = (
                f"Maxvol algorithm did not converge within {_} iterations. "
                f"Current c-max: {cmax}"
            )
            logger.warning(msg)

        self.indices = np.where(flags)[0]
