"""Loss function."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from ase import Atoms
from mpi4py import MPI
from scipy.constants import eV

from motep.calculator import MTP
from motep.initializer import MTPData
from motep.io.mlip.cfg import _get_species


def calc_properties(
    images: list[Atoms],
    comm: MPI.Comm,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fetch energies, forces, and stresses from the training dataset."""
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


def calculate_energy_force_stress(atoms: Atoms):
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = (
        atoms.get_stress(voigt=False)
        if "stress" in atoms.calc.results
        else np.zeros((3, 3))
    )
    return energy, forces, stress


def calc_rmses(
    images: list[Atoms],
    energies: np.ndarray,
    target_energies: np.ndarray,
    forces: np.ndarray,
    target_forces: np.ndarray,
    stresses: np.ndarray,
    target_stresses: np.ndarray,
) -> None:
    """Calculate RMSEs."""
    se_energies = [
        ((energies[i] - target_energies[i]) / len(atoms)) ** 2
        for i, atoms in enumerate(images)
    ]
    total_number_of_atoms = sum(len(atoms) for atoms in images)
    se_forces = [
        np.sum((forces[i] - target_forces[i]) ** 2) / 3.0
        for i, atoms in enumerate(images)
    ]
    se_stress = [
        np.sum((stresses[i] - target_stresses[i]) ** 2) / 9.0
        for i, atoms in enumerate(images)
    ]

    rmse_energy = np.sqrt(np.mean(se_energies))
    rmse_forces = np.sqrt(np.sum(se_forces) / total_number_of_atoms)
    rmse_stress = np.sqrt(np.mean(se_stress))  # eV/Ang^3

    print("RMSE Energy per atom (eV/atom):", rmse_energy)
    print("RMSE force per component (eV/Ang):", rmse_forces)
    print("RMSE stress per component (GPa):", rmse_stress * eV * 1e21)
    print()


class LossFunctionBase(ABC):
    """Loss function."""

    def __init__(
        self,
        images: list[Atoms],
        data: MTPData,
        setting: dict[str, Any],
        *,
        comm: MPI.Comm,
    ) -> None:
        """Loss function.

        Parameters
        ----------
        data : :class:`motep.initializer.MTPData`
            :class:`motep.initializer.MTPData` object.
        images : list[Atoms]
            List of ASE Atoms objects for the training dataset.
        setting : dict[str, Any]
            Setting for the training.
        comm : MPI.Comm
            MPI.Comm object.

        """
        self.images = images
        self.data = data
        self.setting = setting
        self.comm = comm

        species = setting.get("species")
        self.species = list(_get_species(self.images)) if species is None else species

        self.target_energies, self.target_forces, self.target_stresses = (
            calc_properties(self.images, self.comm)
        )

        self.configuration_weight = np.ones(len(self.images))

    @abstractmethod
    def __call__(self, parameters: list[float]) -> float:
        """Evaluate the loss function."""

    def calc_loss_function(self, energies, forces, stresses) -> float:
        # Calculate the energy difference
        energy_ses = (energies - self.target_energies) ** 2
        energy_mse = self.configuration_weight @ energy_ses

        # Calculate the force difference
        force_ses = [
            np.sum((forces[i] - self.target_forces[i]) ** 2)
            for i in range(len(self.target_forces))
        ]
        force_mse = self.configuration_weight @ force_ses

        # Calculate the stress difference
        stress_ses = [
            np.sum((stresses[i] - self.target_stresses[i]) ** 2)
            for i in range(len(self.target_stresses))
        ]
        stress_mse = self.configuration_weight @ stress_ses

        return (
            self.setting["energy-weight"] * energy_mse
            + self.setting["force-weight"] * force_mse
            + self.setting["stress-weight"] * stress_mse
        )

    def calc_rmses(self, parameters: list[float]) -> None:
        energies, forces, stresses = calc_properties(self.images, self.comm)

        calc_rmses(
            self.images,
            energies,
            self.target_energies,
            forces,
            self.target_forces,
            stresses,
            self.target_stresses,
        )


class LossFunction(LossFunctionBase):
    """Loss function."""

    def __init__(self, *args: tuple, engine: str, **kwargs: dict) -> None:
        super().__init__(*args, **kwargs)
        self.engine = engine
        for atoms in self.images:
            atoms.calc = MTP(engine=self.engine, dict_mtp=self.data.data)

        self.configuration_weight = np.ones(len(self.images))

    def __call__(self, parameters: list[float]) -> float:
        for atoms in self.images:
            self.data.update(parameters)
            atoms.calc.update_parameters(self.data.data)
        energies, forces, stresses = calc_properties(self.images, self.comm)
        return self.calc_loss_function(energies, forces, stresses)
