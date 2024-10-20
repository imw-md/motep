"""Tests for Jacobian."""

import copy
import pathlib
import sys

import numpy as np
import pytest

from motep.calculator import MTP
from motep.io.mlip.cfg import read_cfg
from motep.io.mlip.mtp import read_mtp


@pytest.mark.parametrize("coeffs", ["moment_coeffs", "species_coeffs", "radial_coeffs"])
@pytest.mark.parametrize("level", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("engine", ["numpy"])
def test_jac_energy(
    engine: str,
    level: int,
    coeffs: str,
    data_path: pathlib.Path,
) -> None:
    """Test the Jacobian for the energy with respect to the parameters."""
    path = data_path / f"fitting/crystals/multi/{level:02d}"
    if not (path / "pot.mtp").exists():
        pytest.skip()
    atoms = read_cfg(path / "out.cfg", index=-1)
    mtp_data_ref = read_mtp(path / "pot.mtp")
    atoms.calc = MTP(mtp_data=mtp_data_ref, engine=engine)

    atoms.get_potential_energy()
    jac_anl = atoms.calc.engine.jac_energy(atoms)

    dx = 1e-6

    jac_nmr = np.full_like(jac_anl[coeffs], np.nan)

    mtp_data = copy.deepcopy(mtp_data_ref)

    for i in range(jac_nmr.size):
        orig = mtp_data[coeffs].flat[i]

        mtp_data[coeffs].flat[i] = orig + dx
        atoms.calc.update_parameters(mtp_data)
        ep = atoms.get_potential_energy()

        mtp_data[coeffs].flat[i] = orig - dx
        atoms.calc.update_parameters(mtp_data)
        em = atoms.get_potential_energy()

        jac_nmr.flat[i] = (ep - em) / (2.0 * dx)

        mtp_data[coeffs].flat[i] = orig

    print(jac_nmr)
    print(jac_anl[coeffs])

    np.testing.assert_allclose(jac_nmr, jac_anl[coeffs], atol=1e-4)


@pytest.mark.parametrize("coeffs", ["moment_coeffs", "species_coeffs", "radial_coeffs"])
@pytest.mark.parametrize("level", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("engine", ["numpy"])
def test_jac_forces(
    engine: str,
    level: int,
    coeffs: str,
    data_path: pathlib.Path,
) -> None:
    """Test the Jacobian for the forces with respect to the parameters."""
    path = data_path / f"fitting/crystals/multi/{level:02d}"
    if not (path / "pot.mtp").exists():
        pytest.skip()
    atoms = read_cfg(path / "out.cfg", index=-1)
    mtp_data_ref = read_mtp(path / "pot.mtp")
    atoms.calc = MTP(mtp_data=mtp_data_ref, engine=engine)

    atoms.get_potential_energy()
    jac_anl = atoms.calc.engine.jac_forces(atoms)

    dx = 1e-6

    jac_nmr = np.full_like(jac_anl[coeffs], np.nan)

    mtp_data = copy.deepcopy(mtp_data_ref)

    for indices, orig in np.ndenumerate(mtp_data[coeffs]):
        mtp_data[coeffs][indices] = orig + dx
        atoms.calc.update_parameters(mtp_data)
        fp = atoms.get_forces()

        mtp_data[coeffs][indices] = orig - dx
        atoms.calc.update_parameters(mtp_data)
        fm = atoms.get_forces()

        jac_nmr[indices] = (fp - fm) / (2.0 * dx)

        mtp_data[coeffs][indices] = orig

    print(jac_nmr)
    print(jac_anl[coeffs])

    np.testing.assert_allclose(jac_nmr, jac_anl[coeffs], atol=1e-4)
