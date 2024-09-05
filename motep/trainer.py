"""`motep train` command."""

import argparse
import copy
import pathlib
import time
import tomllib
from itertools import product
from typing import Any

import mlippy
import numpy as np
from ase import Atoms
from mpi4py import MPI

from motep.ga import optimization_GA
from motep.io.mlip.cfg import read_cfg
from motep.io.mlip.mtp import read_mtp, write_mtp
from motep.opt import optimization_bfgs, optimization_nelder
from motep.pot import generate_random_numbers
from motep.utils import cd


def make_default_setting() -> dict[str, Any]:
    """Make default setting."""
    return {
        "configurations": "training.cfg",
        "potential_initial": "initial.mtp",
        "potential_final": "final.mtp",
        "energy-weight": 1.0,
        "force-weight": 0.01,
        "stress-weight": 0.0,
        "steps": ["GA", "Nelder-Mead"],
    }


def parse_setting(filename: str) -> dict:
    """Parse setting file."""
    with pathlib.Path(filename).open("rb") as f:
        return tomllib.load(f)


def fetch_target_values(
    images: list[Atoms],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fetch energies, forces, and stresses from the training dataset."""
    energies = [atoms.calc.results["free_energy"] for atoms in images]
    forces = [atoms.calc.results["forces"] for atoms in images]
    stresses = [atoms.calc.results["stress"] for atoms in images]
    return np.array(energies), np.array(forces), np.array(stresses)


def calculate_energy_force_stress(atoms: Atoms, potential):
    atoms.calc = potential
    energy = atoms.get_potential_energy()
    force = atoms.get_forces()
    stress = atoms.get_stress()  # stress property not implemented in morse
    # stress = [0, 0, 0, 0, 0, 0]  # workaround
    return energy, force, stress


def current_value(images: list[Atoms], potential):
    rank = comm.Get_rank()
    size = comm.Get_size()
    # Initialize lists to store energies, forces, and stresses
    current_energies = []
    current_forces = []
    current_stress = []

    if not isinstance(images, list):
        images = [images]

    # Determine the chunk of atoms to process for each MPI process
    chunk_size = len(images) // size
    remainder = len(images) % size
    start_idx = rank * chunk_size
    end_idx = (rank + 1) * chunk_size + (remainder if rank == size - 1 else 0)
    local_atoms = images[start_idx:end_idx]

    # Perform local calculations
    local_results = []
    for atom in local_atoms:
        local_results.append(calculate_energy_force_stress(atom, potential))

    # Gather results from all processes
    all_results = comm.gather(local_results, root=0)

    # Process results on root process
    if rank == 0:
        for result_list in all_results:
            for energy, force, stress in result_list:
                current_energies.append(energy)
                current_forces.append(force)
                current_stress.append(stress)

    # Broadcast the processed results to all processes
    current_energies = comm.bcast(current_energies, root=0)
    current_forces = comm.bcast(current_forces, root=0)
    current_stress = comm.bcast(current_stress, root=0)

    return (
        np.array(current_energies),
        np.array(current_forces),
        np.array(current_stress),
    )


class Fitness:
    """Class for evaluating fitness.

    Parameters
    ----------
    cfg_file : str
        filename containing the training dataset.

    """

    def __init__(self, cfg_file: str, setting: dict[str, Any]):
        self.images = read_cfg(cfg_file, index=":", species=["H"])
        self.setting = setting
        self.target_energies, self.target_forces, self.target_stress = (
            fetch_target_values(self.images)
        )
        self.configuration_weight = np.ones(len(self.images))

    def __call__(self, parameters):
        potential = MTP_field(self.setting["potential_final"], parameters)
        current_energies, current_forces, current_stress = current_value(
            self.images,
            potential,
        )

        # Calculate the energy difference
        energy_ses = (current_energies - self.target_energies) ** 2
        energy_mse = (self.configuration_weight**2) @ energy_ses

        # Calculate the force difference
        force_ses = [
            np.sum((current_forces[i] - self.target_forces[i]) ** 2)
            for i in range(len(self.target_forces))
        ]
        force_mse = (self.configuration_weight**2) @ force_ses

        # Calculate the stress difference
        stress_ses = [
            np.sum((current_stress[i] - self.target_stress[i]) ** 2)
            for i in range(len(self.target_stress))
        ]
        stress_mse = (self.configuration_weight**2) @ stress_ses

        return (
            self.setting["energy-weight"] * energy_mse
            + self.setting["force-weight"] * force_mse
            + self.setting["stress-weight"] * stress_mse
        )


# def RMSE(reference_set, current_set, potential):
#    current_energies, current_forces, current_stress = current_value(current_set, potential)
#    Target_energies, Target_forces, Target_stress = target_value(reference_set)
#
#    error_energy = [((current_energies[i] - Target_energies[i]) / len(current_set[i])) ** 2 for i in range(len(current_set))]
#    error_force = [np.sum(current_forces[i] - Target_forces[i]) / (3 * len(current_set[i])) ** 2 for i in range(len(current_set))]
#    error_stress = [np.sum(current_stress[i] - Target_stress[i]) / 6 ** 2 for i in range(len(current_set))]
#
#    RMSE_energy = (np.sum(error_energy) * 1000) / len(current_set)
#    RMSE_force = (np.sum(error_force)) / len(current_set)
#    RMSE_stress = (np.sum(error_stress)) / len(current_set)
#
#    print("RMSE Energy per atom (meV/atom):", RMSE_energy)
#    print("RMSE force per atom (eV/Ang):", RMSE_force)
#    print("RMSE stress (GPa):", RMSE_force*0.1)


def calc_rmse(cfg, file: str):
    """Calculate RMSEs."""
    ts = mlippy.ase_loadcfgs(cfg)
    mlip = mlippy.initialize()
    mlip = mlippy.mtp()
    mlip.load_potential(file)
    opts = {}
    mlip.add_atomic_type(1)
    potential = mlippy.MLIP_Calculator(mlip, opts)
    errors = mlippy.ase_errors(mlip, ts)
    print(
        "RMSE Energy per atom (meV/atom):",
        1000 * float(errors["Energy per atom: RMS absolute difference"]),
    )
    print(
        "RMSE force per atom (eV/Ang):",
        float(errors["Forces: RMS absolute difference"]),
    )
    print("RMSE stress (GPa):", float(errors["Stresses: RMS absolute difference"]))
    return errors


def MTP_field(file: str, parameters: list[float]):
    data = read_mtp(untrained_mtp)
    data = update_mtp(copy.deepcopy(data), parameters)

    write_mtp(file, data)

    mlip = mlippy.initialize()
    mlip = mlippy.mtp()
    mlip.load_potential(file)
    opts = {}
    mlip.add_atomic_type(1)
    potential = mlippy.MLIP_Calculator(mlip, opts)

    return potential


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
    species_count = data["species_count"]
    species_pairs = list(product(range(species_count), repeat=2))
    asm = data["alpha_scalar_moments"]
    cheb = len(species_pairs) * data["radial_funcs_count"] * data["radial_basis_size"]
    seed = 10
    parameters = (
        [1000]
        + [5] * asm
        + [5] * species_count
        + generate_random_numbers(cheb, -0.1, 0.1, seed)
    )
    bounds = (
        [(-1000, 1000)]
        + [(-5, 5)] * asm
        + [(-5, 5)] * species_count
        + [(-0.1, 0.1)] * cheb
    )
    return parameters, bounds


def update_mtp(
    data: dict[str, Any],
    parameters: list[float],
) -> dict[str, Any]:
    """Update data in the .mtp file.

    Parameters
    ----------
    data : dict[str, Any]
        Data in the .mtp file.
    parameters : list[float]
        MTP parameters.

    Returns
    -------
    data : dict[str, Any]
        Updated data in the .mtp file.

    """
    species_count = data["species_count"]
    rbs = data["radial_basis_size"]
    asm = data["alpha_scalar_moments"]

    data["scaling"] = parameters[0]
    data["moment_coeffs"] = parameters[1 : asm + 1]
    data["species_coeffs"] = parameters[asm + 1 : asm + 1 + species_count]
    total_radial = parameters[asm + 1 + species_count :]
    shape = species_count, species_count, rbs
    total_radial = np.array(total_radial).reshape(shape).tolist()
    data["radial_coeffs"] = {}
    for k0 in range(species_count):
        for k1 in range(species_count):
            data["radial_coeffs"][k0, k1] = [total_radial[k0][k1]]
    return data


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments."""
    parser.add_argument("setting")


def run(args: argparse.Namespace) -> None:
    """Run."""
    global untrained_mtp
    global comm

    start_time = time.time()

    setting = make_default_setting()
    setting.update(parse_setting(args.setting))
    print(setting)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cfg_file = str(pathlib.Path(setting["configurations"]).resolve())
    untrained_mtp = str(pathlib.Path(setting["potential_initial"]).resolve())

    fitness = Fitness(cfg_file, setting)

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
        MTP_field(setting["potential_final"], parameters)
        calc_rmse(cfg_file, setting["potential_final"])

    end_time = time.time()
    print("Total time taken:", end_time - start_time, "seconds")

    comm.Barrier()
    MPI.Finalize()
