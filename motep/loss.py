"""Loss function."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt
from ase import Atoms
from ase.stress import voigt_6_to_full_3x3_stress
from mpi4py import MPI
from scipy.constants import eV

from motep.calculator import MTP
from motep.potentials.mtp.data import MTPData
from motep.setting import LossSetting


def _calc_errors_from_diff(diff: np.ndarray) -> dict[str, float]:
    if diff.size == 0:
        return {"N": diff.size, "MAX": np.nan, "ABS": np.nan, "RMS": np.nan}
    return {
        "N": diff.size,
        "MAX": np.max(np.abs(diff)),
        "ABS": np.mean(np.abs(diff)),
        "RMS": np.sqrt(np.mean(diff**2)),
    }


class LossFunctionEnergy:
    """Energy contribution to the loss function."""

    def __init__(
        self,
        images: list[Atoms],
        *,
        energy_per_atom: bool = False,
        energy_per_conf: bool = True,
    ) -> None:
        """Initialize."""
        self.images = images
        self.energy_per_atom = energy_per_atom
        self.energy_per_conf = energy_per_conf

        numbers_of_atoms = np.fromiter(
            (len(atoms) for atoms in images),
            dtype=float,
            count=len(images),
        )
        self.inverse_numbers_of_atoms = 1.0 / numbers_of_atoms

        self.configuration_weight = np.ones(len(self.images))

    @property
    def target(self) -> np.float64:
        """Return target values."""
        iterable = (atoms.calc.targets["energy"] for atoms in self.images)
        energies = np.fromiter(iterable, dtype=float, count=len(self.images))
        if self.energy_per_atom:
            energies *= self.inverse_numbers_of_atoms
        return energies

    @property
    def result(self) -> np.float64:
        """Return result values."""
        iterable = (atoms.calc.results["energy"] for atoms in self.images)
        energies = np.fromiter(iterable, dtype=float, count=len(self.images))
        if self.energy_per_atom:
            energies *= self.inverse_numbers_of_atoms
        return energies

    def calculate(self) -> np.float64:
        """Calculate the contribution to the loss function."""
        energy_ses = (self.result - self.target) ** 2
        loss = self.configuration_weight @ energy_ses
        return loss / len(self.images) if self.energy_per_conf else loss

    def jac(self) -> npt.NDArray[np.float64]:
        """Calculate the contribution to the loss function Jacobian."""

        def per_configuration(atoms: Atoms) -> np.float64:
            return 2.0 * (
                (atoms.calc.results["energy"] - atoms.calc.targets["energy"])
                * atoms.calc.engine.jac_energy(atoms).parameters
            )

        jacs = np.array([per_configuration(atoms) for atoms in self.images])
        if self.energy_per_atom:
            jacs *= self.inverse_numbers_of_atoms[:, None] ** 2
        jac = self.configuration_weight @ jacs
        return jac / len(self.images) if self.energy_per_conf else jac


class LossFunctionForces:
    """Forces contribution to the loss function.

    Attributes
    ----------
    idcs_frc : npt.NDArray[np.int64]
        Indices of images that have forces.

    """

    def __init__(
        self,
        images: list[Atoms],
        *,
        forces_per_atom: bool = False,
        forces_per_conf: bool = True,
    ) -> None:
        """Initialize."""
        self.images = images
        self.forces_per_atom = forces_per_atom
        self.forces_per_conf = forces_per_conf

        numbers_of_atoms = np.fromiter(
            (len(atoms) for atoms in images),
            dtype=float,
            count=len(images),
        )
        self.inverse_numbers_of_atoms = 1.0 / numbers_of_atoms

        self.idcs_frc = np.fromiter(
            (i for i, atoms in enumerate(images) if "forces" in atoms.calc.results),
            dtype=int,
        )

        self.configuration_weight = np.ones(len(self.images))

    def calculate(self) -> np.float64:
        """Calculate the contribution to the loss function."""
        images = self.images
        key = "forces"
        iterable = (
            np.sum((images[i].calc.results[key] - images[i].calc.targets[key]) ** 2)
            for i in self.idcs_frc
        )
        forces_ses = np.fromiter(iterable, dtype=float, count=self.idcs_frc.size)
        if self.forces_per_atom:
            forces_ses *= self.inverse_numbers_of_atoms[self.idcs_frc]
        loss = self.configuration_weight[self.idcs_frc] @ forces_ses
        return loss / len(self.images) if self.forces_per_conf else loss

    def jac(self) -> npt.NDArray[np.float64]:
        """Calculate the contribution to the loss function Jacobian."""

        def per_configuration(atoms: Atoms) -> np.float64:
            return 2.0 * np.sum(
                (atoms.calc.results["forces"] - atoms.calc.targets["forces"])
                * atoms.calc.engine.jac_forces(atoms).parameters,
                axis=(-2, -1),
            )

        jacs = np.array([per_configuration(self.images[i]) for i in self.idcs_frc])
        if self.forces_per_atom:
            jacs *= self.inverse_numbers_of_atoms[self.idcs_frc, None]
        jac = self.configuration_weight[self.idcs_frc] @ jacs
        return jac / len(self.images) if self.forces_per_conf else jac


class LossFunctionStress:
    """Stress contribution to the loss function.

    Attributes
    ----------
    idcs_str : npt.NDArray[np.int64]
        Indices of images that have 3D cells.

    """

    def __init__(
        self,
        images: list[Atoms],
        *,
        stress_times_volume: bool = False,
        stress_per_conf: bool = True,
    ) -> None:
        """Initialize."""
        self.images = images
        self.stress_times_volume = stress_times_volume
        self.stress_per_conf = stress_per_conf

        self.idcs_str = np.fromiter(
            (i for i, atoms in enumerate(images) if "stress" in atoms.calc.results),
            dtype=int,
        )

        self.volumes = np.fromiter(
            (images[i].cell.volume for i in self.idcs_str),
            dtype=float,
            count=self.idcs_str.size,
        )

        self.configuration_weight = np.ones(len(self.images))

    def calculate(self) -> np.float64:
        """Calculate the contribution to the loss function."""
        images = self.images
        key = "stress"
        f = voigt_6_to_full_3x3_stress
        iterable = (
            np.sum((f(images[i].calc.results[key] - images[i].calc.targets[key])) ** 2)
            for i in self.idcs_str
        )
        stress_ses = np.fromiter(iterable, dtype=float, count=self.idcs_str.size)
        if self.stress_times_volume:
            stress_ses *= self.volumes**2
        loss = self.configuration_weight[self.idcs_str] @ stress_ses
        return loss / len(images) if self.stress_per_conf else loss

    def jac(self) -> npt.NDArray[np.float64]:
        """Calculate the contribution to the loss function Jacobian."""
        f = voigt_6_to_full_3x3_stress

        def per_configuration(atoms: Atoms) -> np.float64:
            return 2.0 * np.sum(
                f(atoms.calc.results["stress"] - atoms.calc.targets["stress"])
                * atoms.calc.engine.jac_stress(atoms).parameters,
                axis=(-2, -1),
            )

        jacs = np.array([per_configuration(self.images[i]) for i in self.idcs_str])
        if self.stress_times_volume:
            jacs *= self.volumes[self.idcs_str, None] ** 2
        jac = self.configuration_weight[self.idcs_str] @ jacs
        return jac / len(self.images) if self.stress_per_conf else jac


class LossFunctionBase(ABC):
    """Loss function."""

    def __init__(
        self,
        images: list[Atoms],
        mtp_data: MTPData,
        setting: LossSetting,
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
        setting : :class:`motep.setting.LossSetting`
            Setting of the loss function.
        comm : MPI.Comm
            MPI.Comm object.

        """
        self.images = images
        self.mtp_data = mtp_data
        self.setting = setting
        self.comm = comm

        self.loss_energy = LossFunctionEnergy(
            self.images,
            energy_per_atom=self.setting.energy_per_atom,
            energy_per_conf=self.setting.energy_per_conf,
        )
        self.loss_forces = LossFunctionForces(
            self.images,
            forces_per_atom=self.setting.forces_per_atom,
            forces_per_conf=self.setting.forces_per_conf,
        )
        self.loss_stress = LossFunctionStress(
            self.images,
            stress_times_volume=self.setting.stress_times_volume,
            stress_per_conf=self.setting.stress_per_conf,
        )

    @abstractmethod
    def __call__(self, parameters: npt.NDArray[np.float64]) -> np.float64:
        """Evaluate the loss function."""

    def _run_calculations(self) -> None:
        """Run calculations of the properties."""
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()
        ncnf = len(self.images)
        for i in range(rank, ncnf, size):
            self.images[i].get_potential_energy()
        for i in range(ncnf):
            results = self.images[i].calc.results
            results.update(self.comm.bcast(results, root=i % size))
            if hasattr(self.images[i].calc, "engine"):
                mbd = self.images[i].calc.engine.mbd
                self.images[i].calc.engine.mbd = self.comm.bcast(mbd, root=i % size)
                rbd = self.images[i].calc.engine.rbd
                self.images[i].calc.engine.rbd = self.comm.bcast(rbd, root=i % size)

    def calc_loss_function(self) -> float:
        """Calculate the value of the loss function."""
        self._run_calculations()
        return (
            self.setting.energy_weight * self.loss_energy.calculate()
            + self.setting.forces_weight * self.loss_forces.calculate()
            + self.setting.stress_weight * self.loss_stress.calculate()
        )

    def jac(self, parameters: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate the Jacobian of the loss function."""
        jac = self.setting.energy_weight * self.loss_energy.jac()
        if self.loss_forces.idcs_frc.size and self.setting.forces_weight:
            jac += self.setting.forces_weight * self.loss_forces.jac()
        if self.loss_stress.idcs_str.size and self.setting.stress_weight:
            jac += self.setting.stress_weight * self.loss_stress.jac()
        return jac


class ErrorPrinter:
    """Printer of errors."""

    def __init__(self, loss: LossFunctionBase) -> None:
        """Initialize `ErrorPrinter`."""
        self.loss = loss

    def _calc_errors_energy(self) -> dict[str, float]:
        loss = self.loss
        iterable = (
            atoms.calc.results["energy"] - atoms.calc.targets["energy"]
            for atoms in loss.images
        )
        return _calc_errors_from_diff(np.fromiter(iterable, dtype=float))

    def _calc_errors_energy_per_atom(self) -> dict[str, float]:
        loss = self.loss
        iterable = (
            ((atoms.calc.results["energy"] - atoms.calc.targets["energy"]) / len(atoms))
            for i, atoms in enumerate(loss.images)
        )
        return _calc_errors_from_diff(np.fromiter(iterable, dtype=float))

    def _calc_errors_forces(self) -> dict[str, float]:
        loss = self.loss
        iterable = (
            loss.images[i].calc.results["forces"][j, k]
            - loss.images[i].calc.targets["forces"][j, k]
            for i in loss.loss_forces.idcs_frc
            for j in range(len(loss.images[i]))
            for k in range(3)
        )
        return _calc_errors_from_diff(np.fromiter(iterable, dtype=float))

    def _calc_errors_stress(self) -> dict[str, float]:
        loss = self.loss
        f = voigt_6_to_full_3x3_stress
        iterable = (
            f(loss.images[i].calc.results["stress"])[j, k]
            - f(loss.images[i].calc.targets["stress"])[j, k]
            for i in loss.loss_stress.idcs_str
            for j in range(3)
            for k in range(3)
        )
        return _calc_errors_from_diff(np.fromiter(iterable, dtype=float))

    def calculate(self) -> dict[str, float]:
        """Calculate errors.

        The properties should be computed before called.

        Returns
        -------
        dict[str, float]
            Errors for the properties.

        """
        errors = {}
        errors["energy"] = self._calc_errors_energy()
        errors["energy_per_atom"] = self._calc_errors_energy_per_atom()
        errors["forces"] = self._calc_errors_forces()
        errors["stress"] = self._calc_errors_stress()  # eV/Ang^3
        return errors

    def print(self, **kwargs: dict[str, Any]) -> dict[str, float]:
        """Print errors.

        `**kwargs` are used to, e.g., give `flush=True` for `print` at the end
        of each block.
        """
        errors = self.calculate()

        key0 = "energy"
        print("Energy (eV):")
        print(f"    Errors checked for {errors[key0]['N']} configurations")
        for key1 in ["MAX", "ABS", "RMS"]:
            print(f"    {key1} error: {errors[key0][key1]}")
        print(**kwargs)

        key0 = "energy_per_atom"
        print("Energy per atom (eV/atom):")
        print(f"    Errors checked for {errors[key0]['N']} configurations")
        for key1 in ["MAX", "ABS", "RMS"]:
            print(f"    {key1} error: {errors[key0][key1]}")
        print(**kwargs)

        key0 = "forces"
        print("Forces per component (eV/angstrom):")
        print(f"    Errors checked for {errors[key0]['N'] // 3} atoms")
        for key1 in ["MAX", "ABS", "RMS"]:
            print(f"    {key1} error: {errors[key0][key1]}")
        print(**kwargs)

        key0 = "stress"
        print("Stress per component (GPa):")
        print(f"    Errors checked for {errors[key0]['N'] // 9} configurations")
        for key1 in ["MAX", "ABS", "RMS"]:
            print(f"    {key1} error: {errors[key0][key1] * eV * 1e21}")
        print(**kwargs)

        return errors


class LossFunction(LossFunctionBase):
    """Loss function."""

    def __init__(self, *args: tuple, engine: str, **kwargs: dict) -> None:
        super().__init__(*args, **kwargs)
        self.engine = engine
        for atoms in self.images:
            targets = atoms.calc.results
            atoms.calc = MTP(self.mtp_data, engine=self.engine, is_trained=True)
            atoms.calc.targets = targets

    def __call__(self, parameters: list[float]) -> float:
        self.mtp_data.parameters = parameters
        for atoms in self.images:
            atoms.calc.update_parameters(self.mtp_data)
        return self.calc_loss_function()
