"""Loss function."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from ase import Atoms
from mpi4py import MPI
from scipy.constants import eV

from motep.calculator import MTP
from motep.potentials import MTPData


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
    local_results = [calculate_energy_force_stress(atoms) for atoms in local_atoms]

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


def calculate_energy_force_stress(atoms: Atoms) -> tuple:
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = (
        atoms.get_stress(voigt=False)
        if "stress" in atoms.calc.results
        else np.zeros((3, 3))
    )
    return energy, forces, stress


def _calc_errors_from_diff(diff: np.ndarray) -> dict[str, float]:
    errors = {}
    errors["N"] = diff.size
    errors["MAX"] = np.max(np.abs(diff))
    errors["ABS"] = np.mean(np.abs(diff))
    errors["RMS"] = np.sqrt(np.mean(diff**2))
    return errors


class LossFunctionBase(ABC):
    """Loss function."""

    def __init__(
        self,
        images: list[Atoms],
        mtp_data: MTPData,
        setting: dict[str, Any],
        *,
        comm: MPI.Comm,
    ) -> None:
        """Loss function.

        Parameters
        ----------
        images : list[Atoms]
            List of ASE Atoms objects for the training dataset.
        mtp_data : :class:`motep.initializer.MTPData`
            :class:`motep.initializer.MTPData` object.
        setting : dict[str, Any]
            Setting for the training.
        comm : MPI.Comm
            MPI.Comm object.

        """
        self.images = images
        self.mtp_data = mtp_data
        self.setting = setting
        self.comm = comm

        self.target_energies, self.target_forces, self.target_stresses = (
            calc_properties(self.images, self.comm)
        )

        self.configuration_weight = np.ones(len(self.images))

    @abstractmethod
    def __call__(self, parameters: list[float]) -> float:
        """Evaluate the loss function."""

    def _calc_loss_energy(self, energies: np.ndarray) -> np.float64:
        energy_ses = (energies - self.target_energies) ** 2  # squared errors
        energy_mse = self.configuration_weight @ energy_ses  # mean squared error
        return self.setting["energy-weight"] * energy_mse

    def _calc_loss_forces(self, forces: list[np.ndarray]) -> np.float64:
        force_ses = [
            np.sum((forces[i] - self.target_forces[i]) ** 2)
            for i in range(len(self.target_forces))
        ]
        force_mse = self.configuration_weight @ force_ses
        return self.setting["force-weight"] * force_mse

    def _calc_loss_stress(self, stresses: list[np.ndarray]) -> np.float64:
        stress_ses = [
            np.sum((stresses[i] - self.target_stresses[i]) ** 2)
            for i in range(len(self.target_stresses))
        ]
        stress_mse = self.configuration_weight @ stress_ses
        return self.setting["stress-weight"] * stress_mse

    def calc_loss_function(self) -> float:
        """Calculate the value of the loss function."""
        energies, forces, stresses = calc_properties(self.images, self.comm)
        loss_energy = self._calc_loss_energy(energies)
        loss_forces = self._calc_loss_forces(forces)
        loss_stress = self._calc_loss_stress(stresses)
        return loss_energy + loss_forces + loss_stress

    def _calc_errors_energy(self, energies: np.ndarray) -> dict[str, float]:
        iterable = (
            energies[i] - self.target_energies[i] for i in range(len(self.images))
        )
        return _calc_errors_from_diff(np.fromiter(iterable, dtype=float))

    def _calc_errors_energy_per_atom(self, energies: np.ndarray) -> dict[str, float]:
        iterable = (
            ((energies[i] - self.target_energies[i]) / len(atoms))
            for i, atoms in enumerate(self.images)
        )
        return _calc_errors_from_diff(np.fromiter(iterable, dtype=float))

    def _calc_errors_forces(self, forces: np.ndarray) -> dict[str, float]:
        iterable = (
            forces[i][j][k] - self.target_forces[i][j][k]
            for i, atoms in enumerate(self.images)
            for j in range(len(atoms))
            for k in range(3)
        )
        return _calc_errors_from_diff(np.fromiter(iterable, dtype=float))

    def _calc_errors_stress(self, stresses: np.ndarray) -> dict[str, float]:
        iterable = (
            stresses[i][j][k] - self.target_stresses[i][j][k]
            for i in range(len(self.images))
            for j in range(3)
            for k in range(3)
        )
        return _calc_errors_from_diff(np.fromiter(iterable, dtype=float))

    def calc_errors(self) -> dict[str, float]:
        """Calculate errors.

        Returns
        -------
        dict[str, float]
            Errors for the properties.

        """
        energies, forces, stresses = calc_properties(self.images, self.comm)
        errors = {}
        errors["energy"] = self._calc_errors_energy(energies)
        errors["energy_per_atom"] = self._calc_errors_energy_per_atom(energies)
        errors["forces"] = self._calc_errors_forces(forces)
        errors["stress"] = self._calc_errors_stress(stresses)  # eV/Ang^3
        return errors

    def print_errors(self) -> dict[str, float]:
        """Print errors."""
        errors = self.calc_errors()

        key0 = "energy"
        print("Energy (eV):")
        print(f"    Errors checked for {errors[key0]['N']} configurations")
        for key1 in ["MAX", "ABS", "RMS"]:
            print(f"    {key1} error: {errors[key0][key1]}")
        print()

        key0 = "energy_per_atom"
        print("Energy per atom (eV/atom):")
        print(f"    Errors checked for {errors[key0]['N']} configurations")
        for key1 in ["MAX", "ABS", "RMS"]:
            print(f"    {key1} error: {errors[key0][key1]}")
        print()

        key0 = "forces"
        print("Forces per component (eV/angstrom):")
        print(f"    Errors checked for {errors[key0]['N'] // 3} atoms")
        for key1 in ["MAX", "ABS", "RMS"]:
            print(f"    {key1} error: {errors[key0][key1]}")
        print()

        key0 = "stress"
        print("Stress per component (GPa):")
        print(f"    Errors checked for {errors[key0]['N'] // 9} configurations")
        for key1 in ["MAX", "ABS", "RMS"]:
            print(f"    {key1} error: {errors[key0][key1] * eV * 1e21}")
        print()

        return errors


class LossFunction(LossFunctionBase):
    """Loss function."""

    def __init__(self, *args: tuple, engine: str, **kwargs: dict) -> None:
        super().__init__(*args, **kwargs)
        self.engine = engine
        for atoms in self.images:
            atoms.calc = MTP(engine=self.engine, dict_mtp=self.mtp_data.dict_mtp)

        self.configuration_weight = np.ones(len(self.images))

    def __call__(self, parameters: list[float]) -> float:
        self.mtp_data.update(parameters)
        for atoms in self.images:
            atoms.calc.update_parameters(self.mtp_data.dict_mtp)
        return self.calc_loss_function()
