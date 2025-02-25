"""Module for algorithms."""

from abc import ABC, abstractmethod
from itertools import combinations
from math import comb

import numpy as np
from ase import Atoms

from motep.calculator import MTP
from motep.potentials.mtp.data import MTPData


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
    ) -> None:
        """Active-set finder."""
        self.images_training = images_training
        self.mtp_data = mtp_data
        self.engine = engine
        self.rng = rng

        self.matrix = self.calc_matrix(self.images_training)
        self.indices = []

        self.find_active_set()

    def calc_matrix(self, images: list[Atoms]) -> np.ndarray:
        """Calculate matrix."""
        # Calculate basis functions of `images_training`
        for atoms in images:
            atoms.calc = MTP(self.mtp_data, engine=self.engine, is_trained=True)
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
        c = matrix @ np.linalg.inv(matrix_active)
        maxvol_grades = np.max(c, axis=1)

        # evaluate `MV_grade` for each `atoms`
        for atoms, maxvol_grade in zip(images, maxvol_grades, strict=True):
            atoms.info["MV_grade"] = maxvol_grade


class ExaustiveAlgorithm(AlgorithmBase):
    """Exaustive algorithm."""

    def find_active_set(self) -> None:
        """Find the active set."""
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


class MaxVolAlgorithm(ExaustiveAlgorithm):
    """MaxVol algorithm."""

    def find_active_set(self) -> None:
        """Find the active set."""
        images = self.images_training
        matrix = self.matrix

        asm = self.mtp_data.alpha_scalar_moments
        indices = self.rng.choice(len(images), asm, replace=False)
        flags = np.zeros(len(images), dtype=bool)
        flags[indices] = True

        tolerance = 1e-9
        for _ in range(65536):  # arbitrary large number for safety
            submatrix = matrix[flags]
            c = matrix @ np.linalg.inv(submatrix)
            i, j = np.divmod(np.argmax(np.abs(c)), asm)
            if np.abs(c[i, j]) < 1.0 + tolerance:
                break
            k = np.where(flags)[0][j]  # row/column in the original matrix
            flags[[k, i]] = flags[[i, k]]
        else:
            raise RuntimeError(_)

        self.indices = np.where(flags)[0]
