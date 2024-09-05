import copy
import os
import time
from itertools import product
from typing import Any

import mlippy
import numpy as np
from ase import Atoms
from mpi4py import MPI

from motep.ga import optimization_GA
from motep.io.mlip.cfg import read_cfg
from motep.io.mlip.mtp import read_mtp, write_mtp
from motep.opt import optimization_nelder
from motep.pot import generate_random_numbers
from motep.utils import cd


def fetch_target_values(
    images: list[Atoms],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fetch energies, forces, and stresses from the training dataset."""
    energies = [atoms.calc.results["free_energy"] for atoms in images]
    forces = [atom.calc.results["forces"] for atom in images]
    stresses = [atom.calc.results["stress"] for atom in images]
    return np.array(energies), np.array(forces), np.array(stresses)


def calculate_energy_force_stress(atom, potential):
    atom.calc = potential
    energy = atom.get_potential_energy()
    force = atom.get_forces()
    stress = atom.get_stress()  # stress property not implemented in morse
    # stress = [0, 0, 0, 0, 0, 0]  # workaround
    return energy, force, stress


def current_value(current_set, potential):
    rank = comm.Get_rank()
    size = comm.Get_size()
    # Initialize lists to store energies, forces, and stresses
    current_energies = []
    current_forces = []
    current_stress = []

    if isinstance(current_set, list):
        atoms = current_set
    else:
        atoms = [current_set]

    # Determine the chunk of atoms to process for each MPI process
    chunk_size = len(atoms) // size
    remainder = len(atoms) % size
    start_idx = rank * chunk_size
    end_idx = (rank + 1) * chunk_size + (remainder if rank == size - 1 else 0)
    local_atoms = atoms[start_idx:end_idx]

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
    """Class for evaluating fitness."""

    def __init__(
        self,
        target_energies,
        target_forces,
        target_stress,
        global_weight,
        configuration_weight,
        current_set,
    ):
        self.target_energies = target_energies
        self.target_forces = target_forces
        self.target_stress = target_stress
        self.global_weight = global_weight
        self.configuration_weight = configuration_weight
        self.current_set = current_set

    def __call__(self, parameters):
        GEW, GFW, GSW = self.global_weight

        # potential = force_field(parameters)
        potential = MTP_field(parameters)
        current_energies, current_forces, current_stress = current_value(
            self.current_set,
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

        return GEW * energy_mse + GFW * force_mse + GSW * stress_mse


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


def MTP_field(parameters: list[float]):
    data = read_mtp(untrained_mtp)
    data = update_mtp(copy.deepcopy(data), parameters)

    file = "Test.mtp"
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
    species_pairs = product(range(species_count), repeat=2)
    w_cheb = species_count + int(data["alpha_scalar_moments"])
    cheb = (
        len(list(species_pairs))
        * int(data["radial_funcs_count"])
        * int(data["radial_basis_size"])
    )
    seed = 10
    parameters = [1000] + [5] * w_cheb + generate_random_numbers(cheb, -0.1, 0.1, seed)
    bounds = [(-1000, 1000)] + [(-5, 5)] * w_cheb + [(-0.1, 0.1)] * cheb
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


def main():
    start_time = time.time()
    current_directory = os.getcwd()
    global untrained_mtp
    global comm

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cfg_file = current_directory + "/final.cfg"
    untrained_mtp = current_directory + "/02.mtp"

    images = read_cfg(cfg_file, index=":", species=["H"])  # training set
    target_energies, target_forces, target_stress = fetch_target_values(images)

    global_weight = [1, 0.01, 0]
    configuration_weight = np.ones(len(images))

    fitness = Fitness(
        target_energies,
        target_forces,
        target_stress,
        global_weight,
        configuration_weight,
        images,
    )

    parameters, bounds = init_parameters(read_mtp(untrained_mtp))

    # Create folders for each rank
    folder_name = f"rank_{rank}"
    folder_path = os.path.join(current_directory, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Change working directory to the created folder
    with cd(folder_path):
        parameters = optimization_GA(fitness, parameters, bounds)
        parameters = optimization_nelder(fitness, parameters, bounds)
        MTP_field(parameters)
        calc_rmse(cfg_file, "Test.mtp")

    end_time = time.time()
    print("Total time taken:", end_time - start_time, "seconds")

    comm.Barrier()
    MPI.Finalize()
