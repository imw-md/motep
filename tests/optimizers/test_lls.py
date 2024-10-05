"""Tests for `LLSOptimizer`."""

import pathlib

import numpy as np
import pytest
from mpi4py import MPI

from motep.io.mlip.cfg import _get_species, read_cfg
from motep.io.mlip.mtp import read_mtp
from motep.loss_function import LossFunction
from motep.optimizers.lls import LLSOptimizer
from motep.potentials import MTPData


@pytest.mark.parametrize("level", [2, 4, 6, 8, 10])
@pytest.mark.parametrize(
    ("molecule", "species"),
    [(762, {1: 0}), (291, {6: 0, 1: 1}), (14214, {9: 0, 1: 1}), (23208, {8: 0})],
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

    species = list(_get_species(images))

    setting = {
        "energy-weight": 1.0,
        "force-weight": 0.01,
        "stress-weight": 0.0,
    }

    mtp_data = MTPData(data, images, species, rng=42)
    parameters, bounds = mtp_data.initialize(optimized=["moment_coeffs"])
    mtp_data.update(parameters)

    loss_function = LossFunction(
        images,
        mtp_data=mtp_data,
        setting=setting,
        comm=MPI.COMM_WORLD,
        engine=engine,
    )

    parameters_ref = np.array(parameters, copy=True)
    mtp_data.print(parameters_ref)
    loss_function.print_errors(parameters_ref)

    optimizer = LLSOptimizer(loss_function)
    parameters = optimizer.optimize(parameters, bounds)
    print()

    mtp_data.print(parameters)
    loss_function.print_errors(parameters)

    # Check if `parameters` are updated.
    assert not np.allclose(parameters, parameters_ref)
