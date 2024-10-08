"""`motep train` command."""

import argparse
import pathlib
import time
from pprint import pprint

from mpi4py import MPI

from motep.io.mlip.cfg import _get_species, read_cfg
from motep.io.mlip.mtp import read_mtp, write_mtp
from motep.loss_function import LossFunction
from motep.optimizers import OptimizerBase
from motep.optimizers.ga import GeneticAlgorithmOptimizer
from motep.optimizers.lls import LLSOptimizer
from motep.optimizers.scipy import (
    ScipyBFGSOptimizer,
    ScipyDifferentialEvolutionOptimizer,
    ScipyDualAnnealingOptimizer,
    ScipyMinimizeOptimizer,
    ScipyNelderMeadOptimizer,
)
from motep.potentials import MTPData
from motep.setting import make_default_setting, parse_setting
from motep.utils import cd


def make_optimizer(optimizer: str) -> OptimizerBase:
    """Make an `Optimizer` class."""
    return {
        "GA": GeneticAlgorithmOptimizer,
        "minimize": ScipyMinimizeOptimizer,
        "Nelder-Mead": ScipyNelderMeadOptimizer,
        "L-BFGS-B": ScipyBFGSOptimizer,
        "DA": ScipyDualAnnealingOptimizer,
        "DE": ScipyDifferentialEvolutionOptimizer,
        "LLS": LLSOptimizer,
    }[optimizer]


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

    dict_mtp = read_mtp(untrained_mtp)

    mtp_data = MTPData(dict_mtp, images, species, setting["seed"])

    if setting["engine"] == "mlippy":
        from motep.mlippy_loss_function import MlippyLossFunction

        fitness = MlippyLossFunction(images, mtp_data, setting, comm=comm)
    else:
        engine = setting["engine"]
        fitness = LossFunction(images, mtp_data, setting, comm=comm, engine=engine)

    # Create folders for each rank
    folder_name = f"rank_{rank}"
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)

    # Change working directory to the created folder
    with cd(folder_name):
        for i, step in enumerate(setting["steps"]):
            print(step["method"])
            print()

            # Print parameters before optimization.
            parameters, bounds = mtp_data.initialize(step["optimized"])
            mtp_data.print(parameters)

            # Instantiate an `Optimizer` class
            optimizer: OptimizerBase = make_optimizer(step["method"])(fitness, **step)

            kwargs = step.get("kwargs", {})
            parameters = optimizer.optimize(parameters, bounds, **kwargs)
            print()

            # Print parameters after optimization.
            mtp_data.update(parameters)
            mtp_data.print(parameters)

            write_mtp(f"intermediate_{i}.mtp", mtp_data.dict_mtp)
            fitness.print_errors()

    mtp_data.update(parameters)
    write_mtp(setting["potential_final"], mtp_data.dict_mtp)

    end_time = time.time()
    print("Total time taken:", end_time - start_time, "seconds")

    comm.Barrier()
    MPI.Finalize()
