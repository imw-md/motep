"""Tests for trainer.py."""

import copy
import pathlib

import numpy as np
import pytest
from mpi4py import MPI

from motep.io.mlip.cfg import read_cfg
from motep.io.mlip.mtp import read_mtp
from motep.loss import LossFunction


@pytest.mark.parametrize("stress_times_volume", [False, True])
@pytest.mark.parametrize("level", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("engine", ["numpy"])
def test_jac(
    *,
    engine: str,
    level: int,
    stress_times_volume: bool,
    data_path: pathlib.Path,
) -> None:
    """Test the Jacobian for the forces with respect to the parameters."""
    path = data_path / f"fitting/crystals/multi/{level:02d}"
    if not (path / "pot.mtp").exists():
        pytest.skip()
    mtp_data = read_mtp(path / "pot.mtp")
    images = read_cfg(path / "out.cfg", index=":")[::1000]

    setting = {
        "energy-weight": 1.0,
        "force-weight": 0.01,
        "stress-weight": 0.001,
        "stress-times-volume": stress_times_volume,
    }

    loss = LossFunction(
        images,
        mtp_data=mtp_data,
        setting=setting,
        comm=MPI.COMM_WORLD,
        engine=engine,
    )
    loss(mtp_data.parameters)
    jac_anl = loss.jac(mtp_data.parameters)

    jac_nmr = np.full_like(jac_anl, np.nan)

    dx = 1e-9

    parameters = mtp_data.parameters

    for i, orig in enumerate(parameters):
        # skip `scaling`
        if i == 0:
            jac_nmr[i] = 0.0
            continue

        parameters[i] = orig + dx
        lp = loss(parameters)

        parameters[i] = orig - dx
        lm = loss(parameters)

        jac_nmr[i] = (lp - lm) / (2.0 * dx)

        parameters[i] = orig

    print(jac_nmr)
    print(jac_anl)

    assert np.any(jac_nmr)  # check if some of the elements are non-zero

    np.testing.assert_allclose(jac_nmr, jac_anl, atol=1e-3)
