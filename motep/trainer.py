"""`motep train` command."""

import argparse
import pathlib
import time
import tomllib
from typing import Any

from mpi4py import MPI

from motep.ga import optimization_GA
from motep.initializer import init_parameters
from motep.io.mlip.mtp import read_mtp
from motep.loss_function import LossFunction
from motep.opt import optimization_bfgs, optimization_nelder
from motep.utils import cd


def make_default_setting() -> dict[str, Any]:
    """Make default setting."""
    return {
        "configurations": "training.cfg",
        "potential_initial": "initial.mtp",
        "potential_final": "final.mtp",
        "optimized": [
            "scaling",
            "radial_coeffs",
            "species_coeffs",
            "moment_coeffs",
        ],
        "seed": None,
        "engine": "numpy",
        "energy-weight": 1.0,
        "force-weight": 0.01,
        "stress-weight": 0.0,
        "steps": ["GA", "Nelder-Mead"],
    }


def parse_setting(filename: str) -> dict:
    """Parse setting file."""
    with pathlib.Path(filename).open("rb") as f:
        return tomllib.load(f)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments."""
    parser.add_argument("setting")


def run(args: argparse.Namespace) -> None:
    """Run."""
    start_time = time.time()

    setting = make_default_setting()
    setting.update(parse_setting(args.setting))
    print(setting)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cfg_file = str(pathlib.Path(setting["configurations"]).resolve())
    untrained_mtp = str(pathlib.Path(setting["potential_initial"]).resolve())

    species = setting.get("species")
    images = read_cfg(cfg_file, index=":", species=species)

    if setting["engine"] == "mlippy":
        from motep.mlippy_loss_function import MlippyLossFunction

        fitness = MlippyLossFunction(images, untrained_mtp, setting, comm)
    else:
        engine = setting["engine"]
        fitness = LossFunction(images, untrained_mtp, setting, comm, engine=engine)

    parameters, bounds = init_parameters(
        read_mtp(untrained_mtp),
        setting["optimized"],
        setting["seed"],
    )

    funs = {
        "GA": optimization_GA,
        "Nelder-Mead": optimization_nelder,
        "L-BFGS-B": optimization_bfgs,
    }

    # Create folders for each rank
    folder_name = f"rank_{rank}"
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)

    # Change working directory to the created folder
    with cd(folder_name):
        for step in setting["steps"]:
            parameters = funs[step](fitness, parameters, bounds)
        fitness.calc_rmses(parameters)

    end_time = time.time()
    print("Total time taken:", end_time - start_time, "seconds")

    comm.Barrier()
    MPI.Finalize()
