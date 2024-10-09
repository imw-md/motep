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
