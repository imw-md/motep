"""`motep grade` command."""

import argparse
import pathlib
import pprint
from abc import ABC, abstractmethod
from itertools import combinations
from math import comb

import numpy as np
from ase import Atoms
from mpi4py import MPI

import motep.io
from motep.calculator import MTP
from motep.io.mlip.mtp import read_mtp
from motep.io.utils import get_dummy_species, read_images
from motep.potentials.mtp.data import MTPData
from motep.setting import load_setting_grade


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments."""
    parser.add_argument("setting")


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

        self.matrix = np.empty((0, 0))
        self.indices = []
        self.find_active_set()

    @abstractmethod
    def find_active_set(self) -> None:
        """Find the active set."""

    def calc_grade(self, images: list[Atoms], engine: str) -> None:
        """Calculate the extrapolation grades."""
        mtp_data = self.mtp_data
        for atoms in images:
            atoms.calc = MTP(mtp_data, engine=engine, is_trained=True)
            atoms.get_potential_energy()

        # Make the overdetermined matrix
        matrix = np.array([atoms.calc.engine.mbd.values for atoms in images])

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

        # Calculate basis functions of `images_training`
        for atoms in images:
            atoms.calc = MTP(self.mtp_data, engine=self.engine, is_trained=True)
            atoms.get_potential_energy()

        # Make the overdetermined matrix
        matrix = np.array([atoms.calc.engine.mbd.values for atoms in images])

        # Choose rows (configurations)
        # This is preliminarily implemented only in an exhausive manner.
        # This is therefore valid so far only for a small `data_in`
        # and for a low level `potential_final`.
        asm = self.mtp_data.alpha_scalar_moments
        if comb(len(images), asm) > 65535:
            msg = "too large possible combinations of rows"
            raise RuntimeError(msg)
        det_max = 0.0
        indices = np.arange(len(images))
        for _ in combinations(range(len(images)), asm):
            indices_checked = np.array(_)
            submatrix = matrix[indices_checked]
            det = np.abs(np.linalg.det(submatrix))
            if det > det_max:
                indices = indices_checked
                det_max = det

        self.matrix = matrix
        self.indices = indices


class MaxVolAlgorithm(ExaustiveAlgorithm):
    """MaxVol algorithm."""

    def find_active_set(self) -> None:
        """Find the active set."""
        images = self.images_training

        # Calculate basis functions of `images_training`
        for atoms in images:
            atoms.calc = MTP(self.mtp_data, engine=self.engine, is_trained=True)
            atoms.get_potential_energy()

        # Make the overdetermined matrix
        matrix = np.array([atoms.calc.engine.mbd.values for atoms in images])

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

        self.matrix = matrix
        self.indices = np.where(flags)[0]


def grade(filename_setting: str, comm: MPI.Comm) -> None:
    """Grade.

    This adds `MV_grade` to `atoms.info`.
    """
    rank = comm.Get_rank()
    setting = load_setting_grade(filename_setting)
    if rank == 0:
        pprint.pp(setting)
        print(flush=True)

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

    mtp_data = read_mtp(mtp_file)
    mtp_data.species = species

    if setting.engine == "mlippy":
        msg = "`mlippy` engine is not available for `motep grade`"
        raise ValueError(msg)

    if setting.algorithm == "exhaustive":
        algorithm_class = ExaustiveAlgorithm
    elif setting.algorithm == "maxvol":
        algorithm_class = MaxVolAlgorithm
    else:
        raise RuntimeError(setting.algorithm)

    optimality = algorithm_class(images_training, mtp_data, setting.engine, rng=rng)

    if rank == 0:
        print(f"{'':=^72s}\n")
        print("[data_active]")
        print(optimality.indices)
        print(flush=True)

    images_in = read_images(
        setting.data_in,
        species=species,
        comm=comm,
        title="data_in",
    )

    optimality.calc_grade(images_in, engine=setting.engine)

    motep.io.write(setting.data_out[0], images_in)


def run(args: argparse.Namespace) -> None:
    """Run."""
    comm = MPI.COMM_WORLD
    grade(args.setting, comm)
