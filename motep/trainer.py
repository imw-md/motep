"""`motep train` command."""

import argparse
import pathlib
import time
from pprint import pprint

from mpi4py import MPI

from motep.io.mlip.cfg import _get_species, read_cfg
from motep.io.mlip.mtp import read_mtp, write_mtp
from motep.loss_function import LossFunction
from motep.optimizers.ga import GeneticAlgorithmOptimizer
from motep.optimizers.lls import LLSOptimizer
from motep.optimizers.scipy import (
    optimize_bfgs,
    optimize_da,
    optimize_de,
    optimize_minimize,
    optimize_nelder,
)
from motep.potentials import MTPData
from motep.setting import make_default_setting, parse_setting
from motep.utils import cd


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments."""
    parser.add_argument("setting")


def run(args: argparse.Namespace) -> None:
    """Run."""
    start_time = time.time()

    setting = make_default_setting()
    setting.update(parse_setting(args.setting))
    pprint(setting, sort_dicts=False)
    print()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cfg_file = str(pathlib.Path(setting["configurations"]).resolve())
    untrained_mtp = str(pathlib.Path(setting["potential_initial"]).resolve())

    species = setting.get("species")
    images = read_cfg(cfg_file, index=":", species=species)
    species = list(_get_species(images)) if species is None else species

    data = read_mtp(untrained_mtp)

    mtp_data = MTPData(data, images, species, setting["seed"])

    if setting["engine"] == "mlippy":
        from motep.mlippy_loss_function import MlippyLossFunction

        fitness = MlippyLossFunction(images, mtp_data, setting, comm=comm)
    else:
        engine = setting["engine"]
        fitness = LossFunction(images, mtp_data, setting, comm=comm, engine=engine)

    # Create folders for each rank
    folder_name = f"rank_{rank}"
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)

    funs = {
        "GA": GeneticAlgorithmOptimizer(data),
        "minimize": optimize_minimize,
        "Nelder-Mead": optimize_nelder,
        "L-BFGS-B": optimize_bfgs,
        "DA": optimize_da,
        "DE": optimize_de,
        "LLS": LLSOptimizer(mtp_data),
    }

    # Change working directory to the created folder
    with cd(folder_name):
        for i, step in enumerate(setting["steps"]):
            parameters, bounds = mtp_data.initialize(step["optimized"])
            mtp_data.print(parameters)
            print(step["method"])
            optimize = funs[step["method"]]
            kwargs = step.get("kwargs", {})
            parameters = optimize(fitness, parameters, bounds, **kwargs)
            mtp_data.update(parameters)
            write_mtp(f"intermediate_{i}.mtp", mtp_data.data)
            print()
            fitness.print_errors(parameters)

    mtp_data.update(parameters)
    write_mtp(setting["potential_final"], mtp_data.data)

    end_time = time.time()
    print("Total time taken:", end_time - start_time, "seconds")

    comm.Barrier()
    MPI.Finalize()
