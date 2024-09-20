"""`motep train` command."""

import argparse
import pathlib
import time
import tomllib
from typing import Any

import numpy as np
from mpi4py import MPI

from motep.ga import optimization_GA
from motep.io.mlip.cfg import read_cfg
from motep.io.mlip.mtp import read_mtp
from motep.loss_function import LossFunction
from motep.opt import optimization_bfgs, optimization_nelder
from motep.pot import generate_random_numbers
from motep.utils import cd


def make_default_setting() -> dict[str, Any]:
    """Make default setting."""
    return {
        "configurations": "training.cfg",
        "potential_initial": "initial.mtp",
        "potential_final": "final.mtp",
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


def init_parameters(
    data: dict[str, Any],
) -> tuple[list[float], list[tuple[float, float]]]:
    """Initialize MTP parameters.

    Parameters
    ----------
    data : dict[str, Any]
        Data in the .mtp file.

    Returns
    -------
    parameters : list[float]
        Initial parameters.
    bounds : list[tuple[float, float]]
        Bounds of the parameters.

    """
    nspecies = data["species_count"]
    asm = data["alpha_scalar_moments"]
    cheb = nspecies**2 * data["radial_funcs_count"] * data["radial_basis_size"]
    seed = 10
    if "scaling" in data:
        tmp = data["scaling"]
        parameters_scaling = [tmp]
        bounds_scaling = [[tmp, tmp]]
    else:
        parameters_scaling = [1000.0]
        bounds_scaling = [(-1000.0, 1000.0)]
    if "moment_coeffs" in data:
        tmp = np.array(data["moment_coeffs"])
        parameters_moment_coeffs = tmp.tolist()
        bounds_moment_coeffs = np.repeat(tmp[:, None], 2, axis=1).tolist()
    else:
        parameters_moment_coeffs = [5.0] * asm
        bounds_moment_coeffs = [(-5.0, 5.0)] * asm
    if "species_coeffs" in data:
        tmp = np.array(data["species_coeffs"])
        parameters_species_coeffs = tmp.tolist()
        bounds_species_coeffs = np.repeat(tmp[:, None], 2, axis=1).tolist()
    else:
        parameters_species_coeffs = [5.0] * nspecies
        bounds_species_coeffs = [(-5.0, 5.0)] * nspecies
    if "radial_coeffs" in data:
        tmp = np.array([p for ps in data["species_coeffs"] for p in ps])
        parameters_radial_coeffs = tmp.tolist()
        bounds_radial_coeffs = np.repeat(tmp[:, None], 2, axis=1).tolist()
    else:
        lb, ub = -0.1, +0.1
        parameters_radial_coeffs = generate_random_numbers(cheb, lb, ub, seed)
        bounds_radial_coeffs = [(lb, ub)] * cheb
    parameters = (
        parameters_scaling
        + parameters_moment_coeffs
        + parameters_species_coeffs
        + parameters_radial_coeffs
    )
    bounds = (
        bounds_scaling
        + bounds_moment_coeffs
        + bounds_species_coeffs
        + bounds_radial_coeffs
    )
    return parameters, bounds


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

    parameters, bounds = init_parameters(read_mtp(untrained_mtp))

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
