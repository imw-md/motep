import copy
from typing import Any

import mlippy
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from mpi4py import MPI

from motep.io.mlip.cfg import _get_species, read_cfg
from motep.io.mlip.mtp import read_mtp, write_mtp
from motep.loss_function import (
    calculate_energy_force_stress,
    fetch_target_values,
    update_mtp,
)


def init_mlip(file: str, species: list[str]):
    mlip = mlippy.initialize()
    mlip = mlippy.mtp()
    mlip.load_potential(file)
    for _ in species:
        mlip.add_atomic_type(chemical_symbols.index(_))
    return mlip


def current_value(images: list[Atoms], comm: MPI.Comm):
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
        local_results.append(calculate_energy_force_stress(atoms))

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


def MTP_field(
    file: str,
    untrained_mtp: str,
    parameters: list[float],
    species: list[str],
):
    data = read_mtp(untrained_mtp)
    data = update_mtp(copy.deepcopy(data), parameters)

    write_mtp(file, data)
    mlip = init_mlip(file, species)
    potential = mlippy.MLIP_Calculator(mlip, {})

    return potential


class MlippyLossFunction:
    """Class for evaluating fitness.

    Parameters
    ----------
    cfg_file : str
        filename containing the training dataset.

    """

    def __init__(
        self,
        cfg_file: str,
        untrained_mtp: str,
        setting: dict[str, Any],
        comm: MPI.Comm,
    ):
        self.untrained_mtp = untrained_mtp
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
        potential = MTP_field(
            file,
            self.untrained_mtp,
            parameters,
            self.species,
        )
        for atoms in self.images:
            atoms.calc = potential
        current_energies, current_forces, current_stress = current_value(
            self.images,
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
        MTP_field(file, self.untrained_mtp, parameters, self.species)

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
