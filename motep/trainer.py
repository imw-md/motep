"""`motep train` command."""

import argparse
import pathlib
import time
from pprint import pprint

from mpi4py import MPI

from motep.initializer import Initializer
from motep.io.mlip.cfg import _get_species, read_cfg
from motep.io.mlip.mtp import read_mtp, write_mtp
from motep.loss_function import LossFunction, update_mtp
from motep.optimizers.ga import optimization_GA
from motep.optimizers.lls import LLSOptimizer
from motep.optimizers.scipy import optimization_bfgs, optimization_nelder
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

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cfg_file = str(pathlib.Path(setting["configurations"]).resolve())
    untrained_mtp = str(pathlib.Path(setting["potential_initial"]).resolve())

    species = setting.get("species")
    images = read_cfg(cfg_file, index=":", species=species)
    species = list(_get_species(images)) if species is None else species

    initializer = Initializer(images, species, setting["seed"])

    if setting["engine"] == "mlippy":
        from motep.mlippy_loss_function import MlippyLossFunction

        fitness = MlippyLossFunction(images, untrained_mtp, setting, comm)
    else:
        engine = setting["engine"]
        fitness = LossFunction(images, untrained_mtp, setting, comm, engine=engine)

    # Create folders for each rank
    folder_name = f"rank_{rank}"
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)

    data = read_mtp(untrained_mtp)

    funs = {
        "GA": optimization_GA,
        "Nelder-Mead": optimization_nelder,
        "L-BFGS-B": optimization_bfgs,
        "LLS": LLSOptimizer(data),
    }

    # Change working directory to the created folder
    with cd(folder_name):
        for i, step in enumerate(setting["steps"]):
            parameters, bounds = initializer.initialize(data, step["optimized"])
            parameters = funs[step["method"]](fitness, parameters, bounds)
            data = update_mtp(data, parameters)
            write_mtp(f"intermediate_{i}.mtp", data)
        fitness.calc_rmses(parameters)

    end_time = time.time()
    print("Total time taken:", end_time - start_time, "seconds")

    comm.Barrier()
    MPI.Finalize()
