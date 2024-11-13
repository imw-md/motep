"""`motep train` command."""

import argparse
import pathlib
import time
from pprint import pprint

import numpy as np
from ase import Atoms
from mpi4py import MPI

from motep.io.mlip.cfg import read_cfg
from motep.io.mlip.mtp import read_mtp, write_mtp
from motep.loss import ErrorPrinter, LossFunction
from motep.optimizers import OptimizerBase, make_optimizer
from motep.setting import parse_setting
from motep.utils import cd


def _get_dummy_species(images: list[Atoms]) -> list[int]:
    m = 0
    for atoms in images:
        m = max(m, atoms.numbers.max())
    return list(range(m + 1))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments."""
    parser.add_argument("setting")


def run(args: argparse.Namespace) -> None:
    """Run."""
    start_time = time.time()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    setting = parse_setting(args.setting)
    if rank == 0:
        pprint(setting, sort_dicts=False)
        print()

    setting.rng = np.random.default_rng(setting.seed)

    cfg_file = str(pathlib.Path(setting.configurations[0]).resolve())
    untrained_mtp = str(pathlib.Path(setting.potential_initial).resolve())

    species = setting.species or None
    images = read_cfg(cfg_file, index=":", species=species)
    if not setting.species:
        setting.species = _get_dummy_species(images)

    mtp_data = read_mtp(untrained_mtp)

    if setting.engine == "mlippy":
        from motep.mlippy_loss import MlippyLossFunction

        loss = MlippyLossFunction(
            images,
            mtp_data,
            setting.loss,
            potential_initial=setting.potential_initial,
            potential_final=setting.potential_final,
            comm=comm,
        )
    else:
        loss = LossFunction(
            images,
            mtp_data,
            setting.loss,
            engine=setting.engine,
            comm=comm,
        )

    # Create folders for each rank
    folder_name = f"rank_{rank}"
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)

    # Change working directory to the created folder
    with cd(folder_name):
        for i, step in enumerate(setting.steps):
            if rank == 0:
                pprint(step, sort_dicts=False)
                print()

            # Print parameters before optimization.
            mtp_data.initialize(setting.rng)
            if rank == 0:
                mtp_data.print()

            # Instantiate an `Optimizer` class
            optimizer: OptimizerBase = make_optimizer(step["method"])(loss, **step)
            optimizer.optimize(**step.get("kwargs", {}))
            if rank == 0:
                print()

            # Print parameters after optimization.
            if rank == 0:
                mtp_data.print()

            write_mtp(f"intermediate_{i}.mtp", mtp_data)
            if rank == 0:
                ErrorPrinter(loss).print()

    write_mtp(setting.potential_final, mtp_data)

    end_time = time.time()
    if rank == 0:
        print("Total time taken:", end_time - start_time, "seconds")

    comm.Barrier()
    MPI.Finalize()
