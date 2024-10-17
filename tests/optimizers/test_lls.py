"""Tests for `LLSOptimizer`."""

import pathlib

import numpy as np
import pytest
from mpi4py import MPI

from motep.io.mlip.cfg import read_cfg
from motep.io.mlip.mtp import read_mtp
from motep.loss_function import LossFunction
from motep.optimizers.lls import LLSOptimizer
from motep.potentials.mtp.data import MTPData


@pytest.mark.parametrize("level", [2, 4, 6, 8, 10])
@pytest.mark.parametrize(
    ("molecule", "species"),
    [(762, [1]), (291, [6, 1]), (14214, [9, 1]), (23208, [8])],
)
@pytest.mark.parametrize("engine", ["numpy"])
def test_molecules(
    engine: str,
    molecule: int,
    species: dict[int, int],
    level: int,
    data_path: pathlib.Path,
) -> None:
    """Test `LLSOptimizer` for molecules."""
    original_path = data_path / f"original/molecules/{molecule}"
    fitting_path = data_path / f"fitting/molecules/{molecule}/{level:02d}"
    if not (fitting_path / "initial.mtp").exists():
        pytest.skip()
    data = read_mtp(fitting_path / "initial.mtp")
    images = read_cfg(original_path / "training.cfg", index=":")

    setting = {
        "energy-weight": 1.0,
        "force-weight": 0.01,
        "stress-weight": 0.0,
    }

    rng = np.random.default_rng(42)

    optimized = ["moment_coeffs"]

    mtp_data = MTPData(data)
    parameters, bounds = mtp_data.initialize(optimized=optimized, rng=rng)
    mtp_data.parameters = parameters
    mtp_data.print()

    loss_function = LossFunction(
        images,
        mtp_data=mtp_data,
        setting=setting,
        comm=MPI.COMM_WORLD,
        engine=engine,
    )

    parameters_ref = np.array(parameters, copy=True)
    loss_function(parameters_ref)  # update paramters
    loss_function.print_errors()

    minimized = ["energy"]
    optimizer = LLSOptimizer(loss_function, optimized=optimized, minimized=minimized)
    parameters = optimizer.optimize(parameters, bounds)
    print()

    mtp_data.parameters = parameters
    mtp_data.print()
    f0 = loss_function(parameters)  # update paramters
    errors0 = loss_function.print_errors()

    # Check if `parameters` are updated.
    assert not np.allclose(parameters, parameters_ref)

    minimized = ["energy", "forces"]
    optimizer = LLSOptimizer(loss_function, optimized=optimized, minimized=minimized)
    parameters = optimizer.optimize(parameters, bounds)
    print()

    mtp_data.parameters = parameters
    mtp_data.print()
    f1 = loss_function(parameters)  # update parameters
    errors1 = loss_function.print_errors()

    # Check loss functions
    # The value should be smaller when considering both energies and forces than
    # when considering only energies.
    assert f0 > f1

    # Check RMSEs
    # When only the RMSE of the energies is minimized, it should be smaller than
    # the value when minimizing the errors of both the energies and the forces.
    assert errors0["energy"]["RMS"] < errors1["energy"]["RMS"]
    assert errors0["forces"]["RMS"] > errors1["forces"]["RMS"]


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
        "force-weight": 0.01,
        "stress-weight": 0.001,
    }

    rng = np.random.default_rng(42)

    optimized = ["moment_coeffs"]

    mtp_data = MTPData(dict_mtp)
    parameters, bounds = mtp_data.initialize(optimized=optimized, rng=rng)
    mtp_data.parameters = parameters

    loss_function = LossFunction(
        images,
        mtp_data=mtp_data,
        setting=setting,
        comm=MPI.COMM_WORLD,
        engine=engine,
    )

    parameters_ref = np.array(parameters, copy=True)
    mtp_data.parameters = parameters
    mtp_data.print()
    loss_function(parameters_ref)  # update parameters
    loss_function.print_errors()

    minimized = ["energy"]
    optimizer = LLSOptimizer(loss_function, optimized=optimized, minimized=minimized)
    parameters = optimizer.optimize(parameters, bounds)
    print()

    mtp_data.parameters = parameters
    mtp_data.print()
    f0 = loss_function(parameters)  # update parameters
    errors0 = loss_function.print_errors()

    # Check if `parameters` are updated.
    assert not np.allclose(parameters, parameters_ref)

    minimized = ["energy", "forces"]
    optimizer = LLSOptimizer(loss_function, optimized=optimized, minimized=minimized)
    parameters = optimizer.optimize(parameters, bounds)
    print()

    mtp_data.parameters = parameters
    mtp_data.print()
    f1 = loss_function(parameters)  # update parameters
    errors1 = loss_function.print_errors()

    # Check loss functions
    # The value should be smaller when considering both energies and forces than
    # when considering only energies.
    assert f0 > f1

    # Check RMSEs
    # When only the RMSE of the energies is minimized, it should be smaller than
    # the value when minimizing the errors of both the energies and the forces.
    assert errors0["energy"]["RMS"] < errors1["energy"]["RMS"]
    assert errors0["forces"]["RMS"] > errors1["forces"]["RMS"]

    minimized = ["energy", "forces", "stress"]
    optimizer = LLSOptimizer(loss_function, optimized=optimized, minimized=minimized)
    parameters = optimizer.optimize(parameters, bounds)
    print()

    mtp_data.parameters = parameters
    mtp_data.print()
    f2 = loss_function(parameters)  # update parameters
    errors2 = loss_function.print_errors()

    # Check loss functions
    assert f1 > f2

    # Check RMSEs
    assert errors1["stress"]["RMS"] > errors2["stress"]["RMS"]


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
    original_path = data_path / f"original/molecules/{molecule}"
    fitting_path = data_path / f"fitting/molecules/{molecule}/{level:02d}"
    if not (fitting_path / "initial.mtp").exists():
        pytest.skip()
    dict_mtp = read_mtp(fitting_path / "initial.mtp")
    images = read_cfg(original_path / "training.cfg", index=":")

    setting = {
        "energy-weight": 1.0,
        "force-weight": 0.01,
        "stress-weight": 0.001,
    }

    rng = np.random.default_rng(42)

    optimized = ["moment_coeffs", "species_coeffs"]
    mtp_data = MTPData(dict_mtp)
    parameters, bounds = mtp_data.initialize(optimized=optimized, rng=rng)
    mtp_data.parameters = parameters

    loss_function = LossFunction(
        images,
        mtp_data=mtp_data,
        setting=setting,
        comm=MPI.COMM_WORLD,
        engine="numpy",
    )

    optimized = ["moment_coeffs"]
    optimizer = LLSOptimizer(loss_function, optimized=optimized, minimized=minimized)
    parameters = optimizer.optimize(parameters, bounds)
    print()

    mtp_data.parameters = parameters
    mtp_data.print()
    f0 = loss_function(parameters)  # update parameters

    optimized = ["moment_coeffs", "species_coeffs"]
    optimizer = LLSOptimizer(loss_function, optimized=optimized, minimized=minimized)
    parameters = optimizer.optimize(parameters, bounds)
    print()

    mtp_data.parameters = parameters
    mtp_data.print()
    f1 = loss_function(parameters)  # update parameters

    # Check loss functions
    assert f0 > f1
