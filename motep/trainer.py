"""`motep train` command."""

import argparse
import pathlib
import sys
import time
from pprint import pprint

import numpy as np
from ase import Atoms
from mpi4py import MPI

from motep.io.mlip.cfg import read_cfg
from motep.io.mlip.mtp import read_mtp, write_mtp
from motep.loss import LossFunction
from motep.optimizers import OptimizerBase, make_optimizer
from motep.setting import make_default_setting, parse_setting
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

    setting = make_default_setting()
    setting.update(parse_setting(args.setting))
    if rank == 0:
        pprint(setting, sort_dicts=False)
        print()

    setting["rng"] = np.random.default_rng(setting["seed"])

    cfg_file = str(pathlib.Path(setting["configurations"]).resolve())
    untrained_mtp = str(pathlib.Path(setting["potential_initial"]).resolve())

    species = setting.get("species")
    images = read_cfg(cfg_file, index=":", species=species)
    if "species" not in setting:
        setting["species"] = _get_dummy_species(images)

    mtp_data = read_mtp(untrained_mtp)

    if setting["engine"] == "mlippy":
        from motep.mlippy_loss import MlippyLossFunction

        loss = MlippyLossFunction(images, mtp_data, setting["loss"], comm=comm)
    else:
        engine = setting["engine"]
        loss = LossFunction(images, mtp_data, setting["loss"], comm=comm, engine=engine)

    # Create folders for each rank
    folder_name = f"rank_{rank}"
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)

    # Change working directory to the created folder
    with cd(folder_name):
        for i, step in enumerate(setting["steps"]):
            if rank == 0:
                print(step["method"])
                print()

            # Print parameters before optimization.
            parameters, bounds = mtp_data.initialize(step["optimized"], setting["rng"])
            mtp_data.parameters = parameters
            if rank == 0:
                mtp_data.print()

            # Instantiate an `Optimizer` class
            optimizer: OptimizerBase = make_optimizer(step["method"])(loss, **step)

            kwargs = step.get("kwargs", {})
            parameters = optimizer.optimize(parameters, bounds, **kwargs)
            if rank == 0:
                print()

            # Print parameters after optimization.
            mtp_data.parameters = parameters
            if rank == 0:
                mtp_data.print()

            write_mtp(f"intermediate_{i}.mtp", mtp_data)
            if rank == 0:
                loss.print_errors()

    mtp_data.parameters = parameters
    write_mtp(setting["potential_final"], mtp_data)

    end_time = time.time()
    if rank == 0:
        print("Total time taken:", end_time - start_time, "seconds")

    comm.Barrier()
    MPI.Finalize()
