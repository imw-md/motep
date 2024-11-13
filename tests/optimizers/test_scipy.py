"""Tests for SciPy-based optimizers."""

import pathlib

import numpy as np
import pytest
from ase import Atoms
from mpi4py import MPI

from motep.io.mlip.cfg import read_cfg
from motep.io.mlip.mtp import read_mtp
from motep.loss import LossFunction
from motep.optimizers.scipy import ScipyMinimizeOptimizer
from motep.potentials.mtp.data import MTPData
from motep.setting import LossSetting


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


def test_without_bounds(data_path: pathlib.Path) -> None:
    """Test if the methods without bounds work."""
    engine = "numpy"
    crystal = "multi"
    level = 2
    images, mtp_data = make_crystals(crystal, level, data_path)
    images = images[::500]

    setting = LossSetting(energy_weight=1.0, forces_weight=0.01, stress_weight=0.001)

    loss = LossFunction(
        images,
        mtp_data=mtp_data,
        setting=setting,
        comm=MPI.COMM_WORLD,
        engine=engine,
    )

    mtp_data.initialize(rng=np.random.default_rng(42))

    kwargs = {"method": "BFGS", "jac": True, "tol": 1e-12, "options": {"maxiter": 10}}

    optimizer = ScipyMinimizeOptimizer(loss)

    optimizer.optimize(**kwargs)


def test_scaling_vs_jac(data_path: pathlib.Path) -> None:
    """Test if `jac=True` and `"scaling" in optimized` raises an error."""
    engine = "numpy"
    crystal = "multi"
    level = 2
    images, mtp_data = make_crystals(crystal, level, data_path)
    images = images[::500]

    setting = LossSetting(energy_weight=1.0, forces_weight=0.01, stress_weight=0.001)

    optimized = ["scaling"]

    loss = LossFunction(
        images,
        mtp_data=mtp_data,
        setting=setting,
        comm=MPI.COMM_WORLD,
        engine=engine,
    )

    mtp_data.optimized = optimized
    mtp_data.initialize(rng=np.random.default_rng(42))

    step = {"method": "L-BFGS-B", "kwargs": {"jac": True, "options": {"maxiter": 10}}}

    optimizer = ScipyMinimizeOptimizer(loss, optimized=optimized, **step)

    with pytest.raises(ValueError, match="scaling"):
        optimizer.optimize(**step.get("kwargs", {}))
