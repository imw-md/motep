"""Tests for `LLSOptimizer`."""

import pathlib

import numpy as np
import pytest
from ase import Atoms
from mpi4py import MPI

from motep.io.mlip.cfg import read_cfg
from motep.io.mlip.mtp import read_mtp
from motep.loss import ErrorPrinter, LossFunction
from motep.optimizers.lls import LLSOptimizer
from motep.setting import LossSetting


def make_molecules(molecule: int, level: int, data_path: pathlib.Path) -> list[Atoms]:
    """Make the ASE `Atoms` object with the calculator."""
    original_path = data_path / f"original/molecules/{molecule}"
    fitting_path = data_path / f"fitting/molecules/{molecule}/{level:02d}"
    if not (fitting_path / "initial.mtp").exists():
        pytest.skip()
    mtp_data = read_mtp(fitting_path / "initial.mtp")
    images = read_cfg(original_path / "training.cfg", index=":")
    return images, mtp_data


@pytest.mark.parametrize("level", [2])
@pytest.mark.parametrize("molecule", [762])
@pytest.mark.parametrize("engine", ["numpy"])
def test_without_forces(
    *,
    engine: str,
    molecule: int,
    level: int,
    data_path: pathlib.Path,
) -> None:
    """Test if `LLSOptimizer` works for the training data without forces."""
    images, mtp_data = make_molecules(molecule, level, data_path)

    for atoms in images:
        del atoms.calc.results["forces"]

    setting = LossSetting(
        energy_weight=1.0,
        forces_weight=0.01,
        stress_weight=0.0,
    )

    rng = np.random.default_rng(42)

    optimized = ["moment_coeffs"]

    parameters, bounds = mtp_data.initialize(optimized=optimized, rng=rng)
    mtp_data.parameters = parameters
    mtp_data.print()

    loss = LossFunction(
        images,
        mtp_data=mtp_data,
        setting=setting,
        comm=MPI.COMM_WORLD,
        engine=engine,
    )

    parameters_ref = np.array(parameters, copy=True)
    loss(parameters_ref)  # update paramters
    ErrorPrinter(loss).print()

    minimized = ["energy", "forces"]
    optimizer = LLSOptimizer(loss, optimized=optimized, minimized=minimized)
    parameters = optimizer.optimize(parameters, bounds)
    print()


@pytest.mark.parametrize("level", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("molecule", [762, 291, 14214, 23208])
@pytest.mark.parametrize("engine", ["numpy", "numba"])
def test_molecules(
    engine: str,
    molecule: int,
    level: int,
    data_path: pathlib.Path,
) -> None:
    """Test `LLSOptimizer` for molecules."""
    images, mtp_data = make_molecules(molecule, level, data_path)

    setting = LossSetting(
        energy_weight=1.0,
        forces_weight=0.01,
        stress_weight=0.0,
    )

    rng = np.random.default_rng(42)

    optimized = ["moment_coeffs"]

    parameters, bounds = mtp_data.initialize(optimized=optimized, rng=rng)
    mtp_data.parameters = parameters
    mtp_data.print()

    loss = LossFunction(
        images,
        mtp_data=mtp_data,
        setting=setting,
        comm=MPI.COMM_WORLD,
        engine=engine,
    )

    parameters_ref = np.array(parameters, copy=True)
    loss(parameters_ref)  # update paramters
    ErrorPrinter(loss).print()

    minimized = ["energy"]
    optimizer = LLSOptimizer(loss, optimized=optimized, minimized=minimized)
    parameters = optimizer.optimize(parameters, bounds)
    print()

    mtp_data.parameters = parameters
    mtp_data.print()
    f0 = loss(parameters)  # update paramters
    errors0 = ErrorPrinter(loss).print()

    # Check if `parameters` are updated.
    assert not np.allclose(parameters, parameters_ref)

    minimized = ["energy", "forces"]
    optimizer = LLSOptimizer(loss, optimized=optimized, minimized=minimized)
    parameters = optimizer.optimize(parameters, bounds)
    print()

    mtp_data.parameters = parameters
    mtp_data.print()
    f1 = loss(parameters)  # update parameters
    errors1 = ErrorPrinter(loss).print()

    # Check loss functions
    # The value should be smaller when considering both energies and forces than
    # when considering only energies.
    assert f0 > f1

    # Check RMSEs
    # When only the RMSE of the energies is minimized, it should be smaller than
    # the value when minimizing the errors of both the energies and the forces.
    assert errors0["energy"]["RMS"] < errors1["energy"]["RMS"]
    assert errors0["forces"]["RMS"] > errors1["forces"]["RMS"]


@pytest.mark.parametrize(
    "optimized",
    [["moment_coeffs"], ["moment_coeffs", "species_coeffs"]],
)
@pytest.mark.parametrize(
    ("energy_per_atom", "stress_times_volume"),
    [(True, False), (False, False), (True, True)],
)
@pytest.mark.parametrize("level", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("crystal", ["cubic", "noncubic"])
@pytest.mark.parametrize("engine", ["numpy", "numba"])
def test_crystals(
    *,
    engine: str,
    crystal: int,
    level: int,
    energy_per_atom: bool,
    stress_times_volume: bool,
    optimized: list[str],
    data_path: pathlib.Path,
) -> None:
    """Test `LLSOptimizer` for crystals."""
    original_path = data_path / f"original/crystals/{crystal}"
    fitting_path = data_path / f"fitting/crystals/{crystal}/{level:02d}"
    if not (fitting_path / "initial.mtp").exists():
        pytest.skip()
    mtp_data = read_mtp(fitting_path / "initial.mtp")
    images = read_cfg(original_path / "training.cfg", index=":")[::100]

    setting = LossSetting(
        energy_weight=1.0,
        forces_weight=0.01,
        stress_weight=0.001,
        energy_per_atom=energy_per_atom,
        stress_times_volume=stress_times_volume,
    )

    rng = np.random.default_rng(42)

    parameters, bounds = mtp_data.initialize(optimized=optimized, rng=rng)
    mtp_data.parameters = parameters

    loss = LossFunction(
        images,
        mtp_data=mtp_data,
        setting=setting,
        comm=MPI.COMM_WORLD,
        engine=engine,
    )

    parameters_ref = np.array(parameters, copy=True)
    mtp_data.parameters = parameters
    mtp_data.print()
    loss(parameters_ref)  # update parameters
    ErrorPrinter(loss).print()

    minimized = ["energy"]
    optimizer = LLSOptimizer(loss, optimized=optimized, minimized=minimized)
    parameters = optimizer.optimize(parameters, bounds)
    print()

    mtp_data.parameters = parameters
    mtp_data.print()
    f0 = loss(parameters)  # update parameters
    errors0 = ErrorPrinter(loss).print()

    # Check if `parameters` are updated.
    assert not np.allclose(parameters, parameters_ref)

    minimized = ["energy", "forces"]
    optimizer = LLSOptimizer(loss, optimized=optimized, minimized=minimized)
    parameters = optimizer.optimize(parameters, bounds)
    print()

    mtp_data.parameters = parameters
    mtp_data.print()
    f1 = loss(parameters)  # update parameters
    errors1 = ErrorPrinter(loss).print()

    # Check RMSEs
    # When only the RMSE of the energies is minimized, it should be smaller than
    # the value when minimizing the errors of both the energies and the forces.
    assert errors0["energy"]["RMS"] < errors1["energy"]["RMS"]
    assert errors0["forces"]["RMS"] > errors1["forces"]["RMS"]

    minimized = ["energy", "forces", "stress"]
    optimizer = LLSOptimizer(loss, optimized=optimized, minimized=minimized)
    parameters = optimizer.optimize(parameters, bounds)
    print()

    mtp_data.parameters = parameters
    mtp_data.print()
    f2 = loss(parameters)  # update parameters
    errors2 = ErrorPrinter(loss).print()

    # Check RMSEs
    assert errors1["stress"]["RMS"] > errors2["stress"]["RMS"]

    # Check loss functions
    # The value should be smaller when all energies, forces, and stress are
    # considered than the value when only part of them are considered.
    assert f0 > f2
    assert f1 > f2


@pytest.mark.parametrize("minimized", [["energy"], ["energy", "forces"]])
@pytest.mark.parametrize("level", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("molecule", [762, 291])
def test_species_coeffs(
    molecule: int,
    level: int,
    minimized: list[str],
    data_path: pathlib.Path,
) -> None:
    """Check if the loss function is smaller when optimizing also `species_coeffs`."""
    images, mtp_data = make_molecules(molecule, level, data_path)

    setting = LossSetting(
        energy_weight=1.0,
        forces_weight=0.01,
        stress_weight=0.001,
    )

    rng = np.random.default_rng(42)

    optimized = ["moment_coeffs", "species_coeffs"]
    parameters, bounds = mtp_data.initialize(optimized=optimized, rng=rng)
    mtp_data.parameters = parameters

    loss = LossFunction(
        images,
        mtp_data=mtp_data,
        setting=setting,
        comm=MPI.COMM_WORLD,
        engine="numpy",
    )

    optimized = ["moment_coeffs"]
    optimizer = LLSOptimizer(loss, optimized=optimized, minimized=minimized)
    parameters = optimizer.optimize(parameters, bounds)
    print()

    mtp_data.parameters = parameters
    mtp_data.print()
    f0 = loss(parameters)  # update parameters

    optimized = ["moment_coeffs", "species_coeffs"]
    optimizer = LLSOptimizer(loss, optimized=optimized, minimized=minimized)
    parameters = optimizer.optimize(parameters, bounds)
    print()

    mtp_data.parameters = parameters
    mtp_data.print()
    f1 = loss(parameters)  # update parameters

    # Check loss functions
    assert f0 > f1
