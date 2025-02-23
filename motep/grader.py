"""`motep grade` command."""

import argparse
import pathlib
import pprint
from itertools import combinations
from math import comb

import numpy as np
from mpi4py import MPI

import motep.io
from motep.io.mlip.mtp import read_mtp
from motep.io.utils import get_dummy_species, read_images
from motep.loss import LossFunction
from motep.setting import parse_setting


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments."""
    parser.add_argument("setting")


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
    images = read_images(setting.data_training, species=species, comm=comm)
    if not setting.species:
        species = get_dummy_species(images)

    mtp_data = read_mtp(mtp_file)
    mtp_data.species = species

    if setting.engine == "mlippy":
        from motep.mlippy_loss import MlippyLossFunction

        loss = MlippyLossFunction(images, mtp_data, setting.loss, comm=comm)
    else:
        loss = LossFunction(
            images,
            mtp_data,
            setting.loss,
            engine=setting.engine,
            comm=comm,
        )

    # Calculate basis functions of `loss.images`
    loss(loss.mtp_data.parameters)

    # Make the overdetermined matrix
    matrix_overdet = np.array([atoms.calc.engine.mbd.values for atoms in loss.images])

    # Choose rows (configurations)
    # This is preliminarily implemented only in an exhausive manner.
    # This is therefore valid so far only for a small number of `data_in` configurations
    # and for a low level `potential_final`.
    asm = mtp_data.alpha_scalar_moments
    if comb(len(loss.images), asm) > 65535:
        msg = "too large possible combinations of rows"
        raise RuntimeError(msg)
    det_max = 0.0
    indices_best = list(range(len(loss.images)))
    for _ in combinations(range(len(loss.images)), asm):
        indices = list(_)
        matrix = matrix_overdet[list(indices)]
        det = np.abs(np.linalg.det(matrix))
        if det > det_max:
            indices_best = list(indices)
            det_max = det

    if rank == 0:
        print("[data_active]")
        print(indices_best)

    # Eq. (8) in [Podryabinkin_CMS_2017_Active]_
    matrix = matrix_overdet[indices_best]
    c = matrix_overdet @ np.linalg.inv(matrix)
    maxvol_grades = np.max(c, axis=1)

    # evaluate `MV_grade` for each `atoms`
    for atoms, maxvol_grade in zip(images, maxvol_grades, strict=True):
        atoms.info["MV_grade"] = maxvol_grade

    motep.io.write(setting.data_out[0], images)


def run(args: argparse.Namespace) -> None:
    """Run."""
    comm = MPI.COMM_WORLD
    grade(args.setting, comm)
