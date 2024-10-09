"""Tests for `Level2MTPOptimizer`."""

import pathlib

import numpy as np
import pytest
from mpi4py import MPI

from motep.io.mlip.cfg import read_cfg
from motep.io.mlip.mtp import read_mtp
from motep.loss_function import LossFunction
from motep.optimizers.ideal import NoInteractionOptimizer
from motep.optimizers.level2mtp import Level2MTPOptimizer
from motep.potentials import MTPData


@pytest.mark.parametrize("level", [2, 4, 6])
@pytest.mark.parametrize("molecule", [762, 291, 14214, 23208])
@pytest.mark.parametrize("engine", ["numpy"])
def test_molecules(
    engine: str,
    molecule: int,
    level: int,
    data_path: pathlib.Path,
) -> None:
    """Test `LLSOptimizer` for molecules."""
    original_path = data_path / f"original/molecules/{molecule}"
    fitting_path = data_path / f"fitting/molecules/{molecule}/{level:02d}"
    if not (fitting_path / "initial.mtp").exists():
        pytest.skip()
    dict_mtp = read_mtp(fitting_path / "initial.mtp")
    species = {762: [1], 291: [6, 1], 14214: [9, 1], 23208: [8]}[molecule]
    dict_mtp["species"] = species
    images = read_cfg(original_path / "training.cfg", index=":", species=species)

    setting = {
        "energy-weight": 1.0,
        "force-weight": 0.01,
        "stress-weight": 0.0,
    }

    mtp_data = MTPData(dict_mtp, rng=42)

    loss = LossFunction(
        images,
        mtp_data=mtp_data,
        setting=setting,
        comm=MPI.COMM_WORLD,
        engine=engine,
    )

    optimizer = NoInteractionOptimizer(loss)
    parameters, bounds = mtp_data.initialize(optimized=[])
    parameters = optimizer.optimize(parameters, bounds)
    print()

    mtp_data.update(parameters)
    mtp_data.print()
    f_ref = loss(parameters)  # update paramters
    loss.print_errors()

    parameters_ref = np.array(parameters, copy=True)

    optimized = ["radial_coeffs"]

    minimized = ["energy"]
    optimizer = Level2MTPOptimizer(loss, optimized=optimized, minimized=minimized)
    parameters = optimizer.optimize(parameters, bounds)
    print()

    mtp_data.update(parameters)
    mtp_data.print()
    f_e00 = loss(parameters)  # update paramters
    errors_e00 = loss.print_errors()

    # Check if `parameters` are updated.
    assert not np.allclose(parameters, parameters_ref)

    # Check loss functions
    assert f_e00 < f_ref


@pytest.mark.parametrize("level", [2, 4, 6, 8, 10])
@pytest.mark.parametrize(
    ("crystal", "species"),
    [("cubic", [29]), ("noncubic", [29])],
)
@pytest.mark.parametrize("engine", ["numpy"])
def test_crystals(
    engine: str,
    crystal: int,
    species: dict[int, int],
    level: int,
    data_path: pathlib.Path,
) -> None:
    """Test PyMTP."""
    original_path = data_path / f"original/crystals/{crystal}"
    fitting_path = data_path / f"fitting/crystals/{crystal}/{level:02d}"
    if not (fitting_path / "initial.mtp").exists():
        pytest.skip()
    dict_mtp = read_mtp(fitting_path / "initial.mtp")
    images = read_cfg(original_path / "training.cfg", index=":")[::100]

    setting = {
        "energy-weight": 1.0,
        "force-weight": 0.0,
        "stress-weight": 0.0,
    }

    optimized = ["radial_coeffs"]

    mtp_data = MTPData(dict_mtp, rng=42)
    parameters, bounds = mtp_data.initialize(optimized=optimized)
    mtp_data.update(parameters)

    loss = LossFunction(
        images,
        mtp_data=mtp_data,
        setting=setting,
        comm=MPI.COMM_WORLD,
        engine=engine,
    )

    optimizer = NoInteractionOptimizer(loss)
    parameters, bounds = mtp_data.initialize(optimized=[])
    parameters = optimizer.optimize(parameters, bounds)
    print()

    mtp_data.update(parameters)
    mtp_data.print()
    loss(parameters)  # update parameters
    loss.print_errors()

    parameters_ref = np.array(parameters, copy=True)

    minimized = ["energy"]
    optimizer = Level2MTPOptimizer(loss, optimized=optimized, minimized=minimized)
    parameters = optimizer.optimize(parameters, bounds)
    print()

    mtp_data.update(parameters)
    mtp_data.print()
    f0 = loss(parameters)  # update parameters
    errors0 = loss.print_errors()

    # Check if `parameters` are updated.
    assert not np.allclose(parameters, parameters_ref)


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
    original_path = data_path / f"original/crystals/{crystal}"
    fitting_path = data_path / f"fitting/crystals/{crystal}/{level:02d}"
    if not (fitting_path / "initial.mtp").exists():
        pytest.skip()
    dict_mtp = read_mtp(fitting_path / "initial.mtp")
    species = [29]
    dict_mtp["species"] = species
    images = read_cfg(original_path / "training.cfg", index=":", species=species)[::100]

    setting = {
        "energy-weight": 1.0,
        "force-weight": 0.0,
        "stress-weight": 0.0,
    }

    optimized = ["radial_coeffs", "species_coeffs"]
    mtp_data = MTPData(dict_mtp, rng=42)
    parameters, bounds = mtp_data.initialize(optimized=optimized)
    mtp_data.update(parameters)

    loss = LossFunction(
        images,
        mtp_data=mtp_data,
        setting=setting,
        comm=MPI.COMM_WORLD,
        engine="numpy",
    )

    optimizer = NoInteractionOptimizer(loss)
    parameters, bounds = mtp_data.initialize(optimized=[])
    parameters = optimizer.optimize(parameters, bounds)
    print()

    mtp_data.update(parameters)
    mtp_data.print()
    loss(parameters)  # update parameters

    optimized = ["radial_coeffs"]
    optimizer = Level2MTPOptimizer(loss, optimized=optimized, minimized=minimized)
    parameters = optimizer.optimize(parameters, bounds)
    print()

    mtp_data.update(parameters)
    mtp_data.print()
    f0 = loss(parameters)  # update parameters

    optimized = ["radial_coeffs", "species_coeffs"]
    optimizer = Level2MTPOptimizer(loss, optimized=optimized, minimized=minimized)
    parameters = optimizer.optimize(parameters, bounds)
    print()

    mtp_data.update(parameters)
    mtp_data.print()
    f1 = loss(parameters)  # update parameters

    # Check loss functions
    assert f0 > f1
