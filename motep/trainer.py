"""`motep train` command."""

import argparse
import copy
import pathlib
import time
import tomllib
from typing import Any

import mlippy
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from mpi4py import MPI

from motep.ga import optimization_GA
from motep.io.mlip.cfg import _get_species, read_cfg
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
    energies = [atoms.get_potential_energy(force_consistent=True) for atoms in images]
    forces = [atoms.get_forces() for atoms in images]
    stresses = [
        atoms.get_stress(voigt=False)
        if "stress" in atoms.calc.results
        else np.zeros((3, 3))
        for atoms in images
    ]
    return np.array(energies), forces, stresses


def calculate_energy_force_stress(atoms: Atoms, potential):
    atoms.calc = potential
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    try:
        stress = atoms.get_stress(voigt=False)
    except NotImplementedError:
        stress = np.zeros((3, 3))
    return energy, forces, stress


def current_value(images: list[Atoms], potential, comm: MPI.Comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    # Initialize lists to store energies, forces, and stresses
    current_energies = []
    current_forces = []
    current_stress = []

    # Determine the chunk of atoms to process for each MPI process
    chunk_size = len(images) // size
    remainder = len(images) % size
    start_idx = rank * chunk_size
    end_idx = (rank + 1) * chunk_size + (remainder if rank == size - 1 else 0)
    local_atoms = images[start_idx:end_idx]

    # Perform local calculations
    local_results = []
    for atoms in local_atoms:
        local_results.append(calculate_energy_force_stress(atoms, potential))

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

    return np.array(current_energies), current_forces, current_stress


def init_mlip(file: str, species: list[str]):
    mlip = mlippy.initialize()
    mlip = mlippy.mtp()
    mlip.load_potential(file)
    for _ in species:
        mlip.add_atomic_type(chemical_symbols.index(_))
    return mlip


class Fitness:
    """Class for evaluating fitness.

    Parameters
    ----------
    cfg_file : str
        filename containing the training dataset.

    """

    def __init__(
        self,
        cfg_file: str,
        setting: dict[str, Any],
        comm: MPI.Comm,
    ):
        species = setting.get("species")
        self.images = read_cfg(cfg_file, index=":", species=species)
        self.species = list(_get_species(self.images)) if species is None else species
        self.setting = setting
        self.target_energies, self.target_forces, self.target_stress = (
            fetch_target_values(self.images)
        )
        self.configuration_weight = np.ones(len(self.images))
        self.comm = comm

    def __call__(self, parameters: list[float]):
        file = self.setting["potential_final"]
        potential = MTP_field(file, parameters, self.species)
        current_energies, current_forces, current_stress = current_value(
            self.images,
            potential,
            self.comm,
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

    def calc_rmse(
        self,
        cfg_file: str,
        parameters: list[float],
    ) -> dict[str, float]:
        """Calculate RMSEs."""
        file = self.setting["potential_final"]
        MTP_field(file, parameters, self.species)

        ts = mlippy.ase_loadcfgs(cfg_file)
        mlip = init_mlip(file, self.species)
        errors = mlippy.ase_errors(mlip, ts)
        print(
            "RMSE Energy per atom (meV/atom):",
            1000 * float(errors["Energy per atom: RMS absolute difference"]),
        )
        print(
            "RMSE force per atom (eV/Ang):",
            float(errors["Forces: RMS absolute difference"]),
        )
        print(
            "RMSE stress (GPa):",
            float(errors["Stresses: RMS absolute difference"]),
        )
        return errors


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


def MTP_field(file: str, parameters: list[float], species: list[str]):
    data = read_mtp(untrained_mtp)
    data = update_mtp(copy.deepcopy(data), parameters)

    write_mtp(file, data)
    mlip = init_mlip(file, species)
    potential = mlippy.MLIP_Calculator(mlip, {})

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

    start_time = time.time()

    setting = make_default_setting()
    setting.update(parse_setting(args.setting))
    print(setting)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cfg_file = str(pathlib.Path(setting["configurations"]).resolve())
    untrained_mtp = str(pathlib.Path(setting["potential_initial"]).resolve())

    fitness = Fitness(cfg_file, setting, comm)

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
        fitness.calc_rmse(cfg_file, parameters)

    end_time = time.time()
    print("Total time taken:", end_time - start_time, "seconds")

    comm.Barrier()
    MPI.Finalize()
