"""Loss function."""

from typing import Any

import numpy as np
from ase import Atoms
from scipy.constants import eV

from motep.calculators import MTP
from motep.io.mlip.cfg import _get_species, read_cfg
from motep.io.mlip.mtp import read_mtp, write_mtp


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


def calculate_energy_force_stress(atoms: Atoms):
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    try:
        stress = atoms.get_stress(voigt=False)
    except NotImplementedError:
        stress = np.zeros((3, 3))
    return energy, forces, stress


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
    rfc = data["radial_funcs_count"]
    rbs = data["radial_basis_size"]
    asm = data["alpha_scalar_moments"]

    data["scaling"] = parameters[0]
    data["moment_coeffs"] = parameters[1 : asm + 1]
    data["species_coeffs"] = parameters[asm + 1 : asm + 1 + species_count]
    total_radial = parameters[asm + 1 + species_count :]
    shape = species_count, species_count, rfc, rbs
    data["radial_coeffs"] = np.array(total_radial).reshape(shape)
    return data


class LossFunction:
    """Loss function."""

    def __init__(
        self,
        cfg_file: str,
        untrained_mtp: str,
        engine: str,
        setting: dict[str, Any],
    ):
        self.untrained_mtp = untrained_mtp
        self.engine = engine
        species = setting.get("species")
        self.images = read_cfg(cfg_file, index=":", species=species)
        self.species = list(_get_species(self.images)) if species is None else species
        self.setting = setting
        self.target_energies, self.target_forces, self.target_stress = (
            fetch_target_values(self.images)
        )
        for atoms in self.images:
            atoms.calc = MTP(
                engine=self.engine,
                mtp_parameters=read_mtp(untrained_mtp),
            )

        self.configuration_weight = np.ones(len(self.images))

    def __call__(self, parameters: list[float]):
        for atoms in self.images:
            data = update_mtp(atoms.calc.engine.parameters, parameters)
            atoms.calc.update_parameters(data)
        current_energies, current_forces, current_stress = self.calc_current_values()

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

    def calc_current_values(self) -> tuple[list, list, list]:
        local_results = []
        for atoms in self.images:
            local_results.append(calculate_energy_force_stress(atoms))

        current_energies = []
        current_forces = []
        current_stress = []
        for energy, force, stress in local_results:
            current_energies.append(energy)
            current_forces.append(force)
            current_stress.append(stress)

        return np.array(current_energies), current_forces, current_stress

    def calc_rmse(self, cfg_file: str, parameters: list[float]) -> None:
        """Calculate RMSEs."""
        self.images = read_cfg(cfg_file, index=":", species=self.species)
        for atoms in self.images:
            atoms.calc = MTP(
                engine=self.engine,
                mtp_parameters=read_mtp(self.untrained_mtp),
            )
            data = update_mtp(atoms.calc.engine.parameters, parameters)
            atoms.calc.update_parameters(data)
        current_energies, current_forces, current_stress = self.calc_current_values()

        se_energies = [
            ((current_energies[i] - self.target_energies[i]) / len(atoms)) ** 2
            for i, atoms in enumerate(self.images)
        ]
        total_number_of_atoms = sum(len(atoms) for atoms in self.images)
        se_forces = [
            np.sum((current_forces[i] - self.target_forces[i]) ** 2)
            for i, atoms in enumerate(self.images)
        ]
        se_stress = [
            np.sum((current_stress[i] - self.target_stress[i]) ** 2) / 9.0
            for i, atoms in enumerate(self.images)
        ]

        rmse_energy = np.sqrt(np.mean(se_energies))  # eV/atom
        rmse_force = np.sqrt(np.sum(se_forces) / total_number_of_atoms)
        rmse_stress = np.sqrt(np.mean(se_stress))  # eV/Ang^3

        print("RMSE Energy per atom (meV/atom):", rmse_energy * 1e3)
        print("RMSE force per atom (eV/Ang):", rmse_force)
        print("RMSE stress (GPa):", rmse_stress * eV * 1e21)

        write_mtp(self.setting["potential_final"], data)
