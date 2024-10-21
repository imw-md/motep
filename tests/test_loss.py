"""Tests for trainer.py."""

import copy
import pathlib

import numpy as np
import pytest
from mpi4py import MPI

from motep.io.mlip.cfg import read_cfg
from motep.io.mlip.mtp import read_mtp
from motep.loss import LossFunction


@pytest.mark.parametrize("level", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("engine", ["numpy"])
def test_jac(engine: str, level: int, data_path: pathlib.Path) -> None:
    """Test the Jacobian for the forces with respect to the parameters."""
    path = data_path / f"fitting/crystals/multi/{level:02d}"
    if not (path / "pot.mtp").exists():
        pytest.skip()
    mtp_data_ref = read_mtp(path / "pot.mtp")
    images = read_cfg(path / "out.cfg", index=":")[::500]

    setting = {
        "energy-weight": 1.0,
        "force-weight": 0.01,
        "stress-weight": 0.001,
    }

    loss = LossFunction(
        images,
        mtp_data=mtp_data_ref,
        setting=setting,
        comm=MPI.COMM_WORLD,
        engine=engine,
    )
    loss(mtp_data_ref.parameters)
    jac_anl = loss.jac()

    jac_nmr = np.full_like(jac_anl, np.nan)

    dx = 1e-6

    mtp_data = copy.deepcopy(mtp_data_ref)

    for i, orig in enumerate(mtp_data.parameters):
        # skip `scaling`
        if i == 0:
            jac_nmr[i] = 0.0
            continue
        mtp_data.parameters[i] = orig + dx
        lp = loss(mtp_data.parameters)

        mtp_data.parameters[i] = orig - dx
        lm = loss(mtp_data.parameters)

        jac_nmr[i] = (lp - lm) / (2.0 * dx)

        mtp_data.parameters[i] = orig

    print(jac_nmr)
    print(jac_anl)

    np.testing.assert_allclose(jac_nmr, jac_anl, atol=1e-4)
