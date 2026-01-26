"""Loss function."""

import logging
from abc import ABC, abstractmethod
from copy import copy

import numpy as np
import numpy.typing as npt
from ase import Atoms
from ase.stress import voigt_6_to_full_3x3_stress
from mpi4py import MPI
from scipy.constants import eV

from motep.calculator import MTP
from motep.potentials.mtp.data import MTPData
from motep.setting import LossSetting

logger = logging.getLogger(__name__)


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
        mtp_data: MTPData,
        energy_per_atom: bool = False,
        energy_per_conf: bool = True,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> None:
        """Initialize."""
        self.images = images
        self.mtp_data = mtp_data
        self.energy_per_atom = energy_per_atom
        self.energy_per_conf = energy_per_conf
        self.comm = comm

        numbers_of_atoms = np.fromiter(
            (len(atoms) for atoms in images),
            dtype=float,
            count=len(images),
        )
        self.inverse_numbers_of_atoms = 1.0 / numbers_of_atoms

        self.configuration_weight = np.ones(len(self.images))

    def calculate(self) -> np.float64:
        """Calculate the contribution to the loss function.

        Returns
        -------
        loss : float
            Energy contribution to the loss function.

        """
        ncnf = len(self.images)
        loss_cnf = 0.0
        for i in range(self.comm.rank, ncnf, self.comm.size):
            atoms = self.images[i]
            target = atoms.calc.targets["energy"]
            result = atoms.calc.results["energy"]
            c = self.configuration_weight[i]
            if self.energy_per_atom:
                c *= self.inverse_numbers_of_atoms[i] ** 2
            loss_cnf += c * (result - target) ** 2
        loss_all = self.comm.allreduce(loss_cnf, op=MPI.SUM)
        return loss_all / ncnf if self.energy_per_conf else loss_all

    def jac(self) -> npt.NDArray[np.float64]:
        """Calculate the contribution to the loss function Jacobian.

        Returns
        -------
        jac : npt.NDArray[np.float64]
            Energy contribution to the loss function Jacobian.

        """
        ncnf = len(self.images)
        nprm = self.mtp_data.number_of_parameters_optimized
        jac_cnf = np.zeros(nprm)
        jac_all = np.zeros(nprm)
        for i in range(self.comm.rank, ncnf, self.comm.size):
            atoms = self.images[i]
            target = atoms.calc.targets["energy"]
            result = atoms.calc.results["energy"]
            c = self.configuration_weight[i]
            if self.energy_per_atom:
                c *= self.inverse_numbers_of_atoms[i] ** 2
            dedp = atoms.calc.engine.jac_energy(atoms).parameters
            jac_cnf += c * 2.0 * (result - target) * dedp
        self.comm.Allreduce(jac_cnf, jac_all, op=MPI.SUM)
        return jac_all / ncnf if self.energy_per_conf else jac_all


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
        mtp_data: MTPData,
        forces_per_atom: bool = False,
        forces_per_conf: bool = True,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> None:
        """Initialize."""
        self.images = images
        self.mtp_data = mtp_data
        self.forces_per_atom = forces_per_atom
        self.forces_per_conf = forces_per_conf
        self.comm = comm

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
        """Calculate the contribution to the loss function.

        Returns
        -------
        loss : float
            Force contribution to the loss function.

        """
        ncnf = len(self.images)
        loss_cnf = 0.0
        for i in range(self.comm.rank, ncnf, self.comm.size):
            if i not in self.idcs_frc:
                continue
            atoms = self.images[i]
            target = atoms.calc.targets["forces"]
            result = atoms.calc.results["forces"]
            c = self.configuration_weight[i]
            if self.forces_per_atom:
                c *= self.inverse_numbers_of_atoms[i]
            loss_cnf += c * np.sum((result - target) ** 2)
        loss_all = self.comm.allreduce(loss_cnf, op=MPI.SUM)
        return loss_all / ncnf if self.forces_per_conf else loss_all

    def jac(self) -> npt.NDArray[np.float64]:
        """Calculate the contribution to the loss function Jacobian.

        Returns
        -------
        jac : npt.NDArray[np.float64]
            Force contribution to the loss function Jacobian.

        """
        ncnf = len(self.images)
        jac_cnf = np.zeros(self.mtp_data.number_of_parameters_optimized)
        jac_all = np.zeros(self.mtp_data.number_of_parameters_optimized)
        for i in range(self.comm.rank, ncnf, self.comm.size):
            if i not in self.idcs_frc:
                continue
            atoms = self.images[i]
            target = atoms.calc.targets["forces"]
            result = atoms.calc.results["forces"]
            c = self.configuration_weight[i]
            if self.forces_per_atom:
                c *= self.inverse_numbers_of_atoms[i]
            dfdp = atoms.calc.engine.jac_forces(atoms).parameters
            jac_cnf += c * 2.0 * np.sum((result - target) * dfdp, axis=(-2, -1))
        self.comm.Allreduce(jac_cnf, jac_all, op=MPI.SUM)
        return jac_all / ncnf if self.forces_per_conf else jac_all


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
        mtp_data: MTPData,
        *,
        stress_times_volume: bool = False,
        stress_per_conf: bool = True,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> None:
        """Initialize."""
        self.images = images
        self.mtp_data = mtp_data
        self.stress_times_volume = stress_times_volume
        self.stress_per_conf = stress_per_conf
        self.comm = comm

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
        """Calculate the contribution to the loss function.

        Returns
        -------
        loss : float
            Stress contribution to the loss function.

        """
        ncnf = len(self.images)
        f = voigt_6_to_full_3x3_stress
        loss_cnf = 0.0
        for i in range(self.comm.rank, ncnf, self.comm.size):
            if i not in self.idcs_str:
                continue
            atoms = self.images[i]
            target = atoms.calc.targets["stress"]
            result = atoms.calc.results["stress"]
            c = self.configuration_weight[i]
            if self.stress_times_volume:
                c *= self.volumes[i] ** 2
            loss_cnf += c * np.sum((f(target - result)) ** 2)
        loss_all = self.comm.allreduce(loss_cnf, op=MPI.SUM)
        return loss_all / ncnf if self.stress_per_conf else loss_all

    def jac(self) -> npt.NDArray[np.float64]:
        """Calculate the contribution to the loss function Jacobian.

        Returns
        -------
        jac : npt.NDArray[np.float64]
            Stress contribution to the loss function Jacobian.

        """
        ncnf = len(self.images)
        f = voigt_6_to_full_3x3_stress
        jac_cnf = np.zeros(self.mtp_data.number_of_parameters_optimized)
        jac_all = np.zeros(self.mtp_data.number_of_parameters_optimized)
        for i in range(self.comm.rank, ncnf, self.comm.size):
            if i not in self.idcs_str:
                continue
            atoms = self.images[i]
            target = atoms.calc.targets["stress"]
            result = atoms.calc.results["stress"]
            c = self.configuration_weight[i]
            if self.stress_times_volume:
                c *= self.volumes[i] ** 2
            dsdp = atoms.calc.engine.jac_stress(atoms).parameters
            jac_cnf += c * 2.0 * np.sum(f(result - target) * dsdp, axis=(-2, -1))
        self.comm.Allreduce(jac_cnf, jac_all, op=MPI.SUM)
        return jac_all / len(self.images) if self.stress_per_conf else jac_all


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

        Notes
        -----
        This class creates a lightweight shallow copy of the provided Atoms
        objects. Atomic positions and arrays are treated as immutable and are
        shared with the input. Only the calculator is replaced internally.

        """
        self.images = [copy(_) for _ in images]
        self.mtp_data = mtp_data
        self.setting = setting
        self.comm = comm

        self.loss_energy = LossFunctionEnergy(
            self.images,
            mtp_data=self.mtp_data,
            energy_per_atom=self.setting.energy_per_atom,
            energy_per_conf=self.setting.energy_per_conf,
            comm=self.comm,
        )
        self.loss_forces = LossFunctionForces(
            self.images,
            mtp_data=self.mtp_data,
            forces_per_atom=self.setting.forces_per_atom,
            forces_per_conf=self.setting.forces_per_conf,
            comm=self.comm,
        )
        self.loss_stress = LossFunctionStress(
            self.images,
            mtp_data=self.mtp_data,
            stress_times_volume=self.setting.stress_times_volume,
            stress_per_conf=self.setting.stress_per_conf,
            comm=self.comm,
        )

    @abstractmethod
    def __call__(self, parameters: npt.NDArray[np.float64]) -> np.float64:
        """Evaluate the loss function."""

    def _run_calculations(self) -> None:
        """Run calculations of the properties."""
        ncnf = len(self.images)
        for i in range(self.comm.rank, ncnf, self.comm.size):
            self.images[i].get_potential_energy()

    def broadcast_results(self) -> None:
        """Broadcast data."""
        size = self.comm.size
        ncnf = len(self.images)
        for i in range(ncnf):
            results = self.images[i].calc.results
            results.update(self.comm.bcast(results, root=i % size))

    def gather_data(self) -> None:
        """Gather data to root process."""
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()
        ncnf = len(self.images)
        if rank == 0:
            for i in range(ncnf):
                root = i % size
                if root != 0 and hasattr(self.images[i].calc, "engine"):
                    mbd = self.comm.recv(source=root, tag=i + ncnf)
                    self.images[i].calc.engine.mbd = mbd
                    rbd = self.comm.recv(source=root, tag=i + 2 * ncnf)
                    self.images[i].calc.engine.rbd = rbd
        else:
            for i in range(rank, ncnf, size):
                if hasattr(self.images[i].calc, "engine"):
                    self.comm.send(self.images[i].calc.engine.mbd, dest=0, tag=i + ncnf)
                    self.comm.send(
                        self.images[i].calc.engine.rbd, dest=0, tag=i + 2 * ncnf
                    )

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

    def log(self) -> dict[str, float]:
        """Log errors.

        Returns
        -------
        errors : dict[str, float]
            Errors.

        """
        errors = self.calculate()

        key0 = "energy"
        logger.info("Energy (eV):")
        logger.info(f"    Errors checked for {errors[key0]['N']} configurations")
        for key1 in ["MAX", "ABS", "RMS"]:
            logger.info(f"    {key1} error: {errors[key0][key1]}")
        logger.info("")

        key0 = "energy_per_atom"
        logger.info("Energy per atom (eV/atom):")
        logger.info(f"    Errors checked for {errors[key0]['N']} configurations")
        for key1 in ["MAX", "ABS", "RMS"]:
            logger.info(f"    {key1} error: {errors[key0][key1]}")
        logger.info("")

        key0 = "forces"
        logger.info("Forces per component (eV/angstrom):")
        logger.info(f"    Errors checked for {errors[key0]['N'] // 3} atoms")
        for key1 in ["MAX", "ABS", "RMS"]:
            logger.info(f"    {key1} error: {errors[key0][key1]}")
        logger.info("")

        key0 = "stress"
        logger.info("Stress per component (GPa):")
        logger.info(f"    Errors checked for {errors[key0]['N'] // 9} configurations")
        for key1 in ["MAX", "ABS", "RMS"]:
            logger.info(f"    {key1} error: {errors[key0][key1] * eV * 1e21}")
        logger.info("")

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
