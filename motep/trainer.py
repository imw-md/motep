"""`motep train` command."""

import argparse
import pathlib
import pprint

import numpy as np
from ase import Atoms
from mpi4py import MPI

from motep.io.mlip.cfg import read_cfg
from motep.io.mlip.mtp import read_mtp, write_mtp
from motep.loss import ErrorPrinter, LossFunction
from motep.optimizers import OptimizerBase, make_optimizer
from motep.setting import parse_setting
from motep.utils import measure_time


def _get_dummy_species(images: list[Atoms]) -> list[int]:
    m = 0
    for atoms in images:
        m = max(m, atoms.numbers.max())
    return list(range(m + 1))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments."""
    parser.add_argument("setting")


def train(filename_setting: str, comm: MPI.Comm) -> None:
    rank = comm.Get_rank()

    setting = parse_setting(filename_setting)
    if rank == 0:
        pprint.pp(setting)
        print(flush=True)

    setting.rng = np.random.default_rng(setting.seed)

    cfg_file = str(pathlib.Path(setting.configurations[0]).resolve())
    untrained_mtp = str(pathlib.Path(setting.potential_initial).resolve())

    species = setting.species or None
    images = read_cfg(cfg_file, index=":", species=species)
    if not setting.species:
        species = _get_dummy_species(images)

    mtp_data = read_mtp(untrained_mtp)
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

    for i, step in enumerate(setting.steps):
        with measure_time(f"step {i}: {step['method']}", comm):
            if rank == 0:
                pprint.pp(step)
                print(flush=True)

            # Print parameters before optimization.
            mtp_data.initialize(setting.rng)
            if rank == 0:
                mtp_data.print()

            # Instantiate an `Optimizer` class
            optimizer: OptimizerBase = make_optimizer(step["method"])(loss, **step)
            optimizer.optimize(**step.get("kwargs", {}))
            if rank == 0:
                print(flush=True)

                # Print parameters after optimization.
                mtp_data.print(flush=True)

                write_mtp(f"intermediate_{i}.mtp", mtp_data)

                ErrorPrinter(loss).print(flush=True)

    if rank == 0:
        write_mtp(setting.potential_final, mtp_data)


def run(args: argparse.Namespace) -> None:
    """Run."""
    comm = MPI.COMM_WORLD
    with measure_time("total"):
        train(args.setting, comm)
