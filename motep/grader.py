"""`motep grade` command."""

import argparse
import pathlib
import pprint
from itertools import combinations
from math import comb

import numpy as np
from ase import Atoms
from mpi4py import MPI

import motep.io
from motep.calculator import MTP
from motep.io.mlip.mtp import read_mtp
from motep.io.utils import get_dummy_species, read_images
from motep.loss import LossFunction, LossFunctionBase
from motep.setting import parse_setting


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments."""
    parser.add_argument("setting")


class DOptimality:
    """Active-set finder.

    Attributes
    ----------
    indices_best : list[int]
        Indices for `loss.images` that consist of the active set.

    """

    def __init__(self, loss: LossFunctionBase) -> None:
        """Active-set finder."""
        self.loss = loss
        self.matrix_overdet = np.empty((0, 0))
        self.indices_best = []

    def find_active_set(self) -> None:
        """Find the active set."""
        images = self.loss.images

        # Make the overdetermined matrix
        matrix_overdet = np.array([atoms.calc.engine.mbd.values for atoms in images])

        # Choose rows (configurations)
        # This is preliminarily implemented only in an exhausive manner.
        # This is therefore valid so far only for a small `data_in`
        # and for a low level `potential_final`.
        asm = self.loss.mtp_data.alpha_scalar_moments
        if comb(len(images), asm) > 65535:
            msg = "too large possible combinations of rows"
            raise RuntimeError(msg)
        det_max = 0.0
        indices_best = list(range(len(images)))
        for _ in combinations(range(len(images)), asm):
            indices = list(_)
            matrix = matrix_overdet[list(indices)]
            det = np.abs(np.linalg.det(matrix))
            if det > det_max:
                indices_best = list(indices)
                det_max = det

        self.matrix_overdet = matrix_overdet
        self.indices_best = indices_best

    def calc_grade(self, images: list[Atoms], engine: str) -> None:
        """Calculate the extrapolation grades."""
        mtp_data = self.loss.mtp_data
        for atoms in images:
            atoms.calc = MTP(mtp_data, engine=engine, is_trained=True)
            atoms.get_potential_energy()

        # Make the overdetermined matrix
        matrix = np.array([atoms.calc.engine.mbd.values for atoms in images])

        # Eq. (8) in [Podryabinkin_CMS_2017_Active]_
        matrix_active = self.matrix_overdet[self.indices_best]
        c = matrix @ np.linalg.inv(matrix_active)
        maxvol_grades = np.max(c, axis=1)

        # evaluate `MV_grade` for each `atoms`
        for atoms, maxvol_grade in zip(images, maxvol_grades, strict=True):
            atoms.info["MV_grade"] = maxvol_grade


def grade(filename_setting: str, comm: MPI.Comm) -> None:
    """Grade.

    This adds `MV_grade` to `atoms.info`.
    """
    rank = comm.Get_rank()
    setting = parse_setting(filename_setting)
    if rank == 0:
        pprint.pp(setting)
        print(flush=True)

    setting.rng = np.random.default_rng(setting.seed)

    mtp_file = str(pathlib.Path(setting.potential_final).resolve())

    species = setting.species or None
    images_training = read_images(setting.data_training, species=species, comm=comm)
    if not setting.species:
        species = get_dummy_species(images_training)

    mtp_data = read_mtp(mtp_file)
    mtp_data.species = species

    if setting.engine == "mlippy":
        msg = "`mlippy` engine is not available for `motep grade"
        raise ValueError(msg)

    loss = LossFunction(
        images_training,
        mtp_data,
        setting.loss,
        engine=setting.engine,
        comm=comm,
    )

    # Calculate basis functions of `loss.images`
    loss(loss.mtp_data.parameters)

    optimality = DOptimality(loss)
    optimality.find_active_set()

    if rank == 0:
        print("[data_active]")
        print(optimality.indices_best)

    images_in = read_images(setting.data_in, species=species, comm=comm)

    optimality.calc_grade(images_in, engine=setting.engine)

    motep.io.write(setting.data_out[0], images_in)


def run(args: argparse.Namespace) -> None:
    """Run."""
    comm = MPI.COMM_WORLD
    grade(args.setting, comm)
