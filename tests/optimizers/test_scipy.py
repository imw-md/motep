"""Tests for SciPy-based optimizers."""

import pathlib

import numpy as np
import pytest
from mpi4py import MPI

from motep.io.mlip.cfg import read_cfg
from motep.io.mlip.mtp import read_mtp
from motep.loss import LossFunction
from motep.optimizers.scipy import ScipyMinimizeOptimizer


def test_scaling_vs_jac(data_path: pathlib.Path) -> None:
    """Test if `jac=True` and `"scaling" in optimized` raises an error."""
    engine = "numpy"
    crystal = "multi"
    level = 2
    original_path = data_path / f"original/crystals/{crystal}"
    fitting_path = data_path / f"fitting/crystals/{crystal}/{level:02d}"
    if not (fitting_path / "initial.mtp").exists():
        pytest.skip()
    mtp_data = read_mtp(fitting_path / "initial.mtp")
    images = read_cfg(original_path / "training.cfg", index=":")[::500]

    setting = {
        "energy-weight": 1.0,
        "force-weight": 0.01,
        "stress-weight": 0.001,
        "steps": [
            {"method": "L-BFGS-B", "kwargs": {"jac": True, "options": {"maxiter": 10}}},
        ],
    }

    optimized = ["scaling"]

    loss = LossFunction(
        images,
        mtp_data=mtp_data,
        setting=setting,
        comm=MPI.COMM_WORLD,
        engine=engine,
    )

    parameters, bounds = mtp_data.initialize(
        optimized=optimized,
        rng=np.random.default_rng(42),
    )

    step = setting["steps"][0]

    optimizer = ScipyMinimizeOptimizer(loss, optimized=optimized, **step)

    with pytest.raises(ValueError, match="scaling"):
        parameters = optimizer.optimize(parameters, bounds, **step.get("kwargs", {}))
