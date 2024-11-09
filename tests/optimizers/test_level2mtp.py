"""Tests for `Level2MTPOptimizer`."""

import pathlib

import numpy as np
import pytest
from ase import Atoms
from mpi4py import MPI

from motep.io.mlip.cfg import read_cfg
from motep.io.mlip.mtp import read_mtp
from motep.loss import ErrorPrinter, LossFunction
from motep.optimizers.ideal import NoInteractionOptimizer
from motep.optimizers.level2mtp import Level2MTPOptimizer
from motep.potentials.mtp.data import MTPData
from motep.setting import LossSetting


def make_molecules(
    molecule: int,
    level: int,
    data_path: pathlib.Path,
) -> tuple[list[Atoms], MTPData]:
    """Make the ASE `Atoms` object with the calculator."""
    original_path = data_path / f"original/molecules/{molecule}"
    fitting_path = data_path / f"fitting/molecules/{molecule}/{level:02d}"
    if not (fitting_path / "initial.mtp").exists():
        pytest.skip()
    mtp_data = read_mtp(fitting_path / "initial.mtp")
    images = read_cfg(original_path / "training.cfg", index=":")
    return images, mtp_data


def make_crystals(
    crystal: str,
    level: int,
    data_path: pathlib.Path,
) -> tuple[list[Atoms], MTPData]:
    """Make the ASE `Atoms` object with the calculator."""
    original_path = data_path / f"original/crystals/{crystal}"
    fitting_path = data_path / f"fitting/crystals/{crystal}/{level:02d}"
    if not (fitting_path / "initial.mtp").exists():
        pytest.skip()
    mtp_data = read_mtp(fitting_path / "initial.mtp")
    images = read_cfg(original_path / "training.cfg", index=":")
    return images, mtp_data


def test_without_forces(data_path: pathlib.Path) -> None:
    """Test if `Level2MTPOptimizer` works for the training data without forces."""
    engine = "numpy"
    molecule = 762
    level = 2
    images, mtp_data = make_molecules(molecule, level, data_path)

    for atoms in images:
        del atoms.calc.results["forces"]

    setting = LossSetting(
        energy_weight=1.0,
        forces_weight=0.01,
        stress_weight=0.0,
    )

    rng = np.random.default_rng(42)

    loss = LossFunction(
        images,
        mtp_data=mtp_data,
        setting=setting,
        comm=MPI.COMM_WORLD,
        engine=engine,
    )

    optimized = ["radial_coeffs"]

    minimized = ["energy", "forces"]
    optimizer = Level2MTPOptimizer(loss, optimized=optimized, minimized=minimized)
    parameters, bounds = mtp_data.initialize(optimized=[], rng=rng)
    parameters = optimizer.optimize(parameters, bounds)
    print()


@pytest.mark.parametrize("level", [2, 4, 6])
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

    loss = LossFunction(
        images,
        mtp_data=mtp_data,
        setting=setting,
        comm=MPI.COMM_WORLD,
        engine=engine,
    )

    optimizer = NoInteractionOptimizer(loss)
    parameters, bounds = mtp_data.initialize(optimized=[], rng=rng)
    parameters = optimizer.optimize(parameters, bounds)
    print()

    mtp_data.parameters = parameters
    mtp_data.print()
    f_ref = loss(parameters)  # update paramters
    ErrorPrinter(loss).print()

    parameters_ref = np.array(parameters, copy=True)

    optimized = ["radial_coeffs"]

    minimized = ["energy"]
    optimizer = Level2MTPOptimizer(loss, optimized=optimized, minimized=minimized)
    parameters = optimizer.optimize(parameters, bounds)
    print()

    mtp_data.parameters = parameters
    mtp_data.print()
    f_e00 = loss(parameters)  # update paramters
    ErrorPrinter(loss).print()

    # Check if `parameters` are updated.
    assert not np.allclose(parameters, parameters_ref)

    # Check loss functions
    assert f_e00 < f_ref


@pytest.mark.parametrize(
    "optimized",
    [
        ["radial_coeffs"],
        ["radial_coeffs", "species_coeffs"],
    ],
)
@pytest.mark.parametrize(
    (
        "energy_per_atom",
        "forces_per_atom",
        "stress_times_volume",
        "energy_per_conf",
        "forces_per_conf",
        "stress_per_conf",
    ),
    [
        (True, True, False, True, True, True),  # default
        (False, True, False, True, True, True),
        (True, False, False, True, True, True),
        (True, True, True, True, True, True),
        (True, True, False, False, True, True),
        (True, True, False, True, False, True),
        (True, True, False, True, True, False),
    ],
)
@pytest.mark.parametrize("level", [2, 4])
@pytest.mark.parametrize("crystal", ["cubic", "noncubic"])
@pytest.mark.parametrize("engine", ["numpy"])
def test_crystals(
    *,
    engine: str,
    crystal: int,
    level: int,
    energy_per_atom: bool,
    forces_per_atom: bool,
    stress_times_volume: bool,
    energy_per_conf: bool,
    forces_per_conf: bool,
    stress_per_conf: bool,
    optimized: list[str],
    data_path: pathlib.Path,
) -> None:
    """Test `Level2MTPOptimizer` for crystals."""
    images, mtp_data = make_crystals(crystal, level, data_path)
    images = images[::100]

    setting = LossSetting(
        energy_weight=1.0,
        forces_weight=0.01,
        stress_weight=0.001,
        energy_per_atom=energy_per_atom,
        forces_per_atom=forces_per_atom,
        stress_times_volume=stress_times_volume,
        energy_per_conf=energy_per_conf,
        forces_per_conf=forces_per_conf,
        stress_per_conf=stress_per_conf,
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

    optimizer = NoInteractionOptimizer(loss)
    parameters, bounds = mtp_data.initialize(optimized=[], rng=rng)
    parameters = optimizer.optimize(parameters, bounds)
    print()

    mtp_data.parameters = parameters
    mtp_data.print()
    loss(parameters)  # update parameters
    ErrorPrinter(loss).print()

    parameters_ref = np.array(parameters, copy=True)

    minimized = ["energy"]
    optimizer = Level2MTPOptimizer(loss, optimized=optimized, minimized=minimized)
    parameters = optimizer.optimize(parameters, bounds)
    print()

    mtp_data.parameters = parameters
    mtp_data.print()
    f0 = loss(parameters)  # update parameters
    errors0 = ErrorPrinter(loss).print()

    # Check if `parameters` are updated.
    assert not np.allclose(parameters, parameters_ref)

    minimized = ["energy", "forces"]
    optimizer = Level2MTPOptimizer(loss, optimized=optimized, minimized=minimized)
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
    optimizer = Level2MTPOptimizer(loss, optimized=optimized, minimized=minimized)
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
    # Note that f0 > f1 is not always true because of the stress contribution.
    assert f0 > f2
    assert f1 > f2


@pytest.mark.parametrize("minimized", [["energy"]])
@pytest.mark.parametrize("level", [2, 4])
@pytest.mark.parametrize("crystal", ["cubic", "noncubic"])
def test_species_coeffs(
    crystal: int,
    level: int,
    minimized: list[str],
    data_path: pathlib.Path,
) -> None:
    """Check if the loss function is smaller when optimizing also `species_coeffs`."""
    images, mtp_data = make_crystals(crystal, level, data_path)
    images = images[::100]

    setting = LossSetting(
        energy_weight=1.0,
        forces_weight=0.0,
        stress_weight=0.0,
    )

    rng = np.random.default_rng(42)

    optimized = ["radial_coeffs", "species_coeffs"]
    parameters, bounds = mtp_data.initialize(optimized=optimized, rng=rng)
    mtp_data.parameters = parameters

    loss = LossFunction(
        images,
        mtp_data=mtp_data,
        setting=setting,
        comm=MPI.COMM_WORLD,
        engine="numpy",
    )

    optimizer = NoInteractionOptimizer(loss)
    parameters, bounds = mtp_data.initialize(optimized=[], rng=rng)
    parameters = optimizer.optimize(parameters, bounds)
    print()

    mtp_data.parameters = parameters
    mtp_data.print()
    loss(parameters)  # update parameters

    optimized = ["radial_coeffs"]
    optimizer = Level2MTPOptimizer(loss, optimized=optimized, minimized=minimized)
    parameters = optimizer.optimize(parameters, bounds)
    print()

    mtp_data.parameters = parameters
    mtp_data.print()
    f0 = loss(parameters)  # update parameters

    optimized = ["radial_coeffs", "species_coeffs"]
    optimizer = Level2MTPOptimizer(loss, optimized=optimized, minimized=minimized)
    parameters = optimizer.optimize(parameters, bounds)
    print()

    mtp_data.parameters = parameters
    mtp_data.print()
    f1 = loss(parameters)  # update parameters

    # Check loss functions
    assert f0 > f1
