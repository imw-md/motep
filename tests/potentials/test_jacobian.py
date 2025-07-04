"""Tests for Jacobian."""

import copy
import pathlib
from typing import TYPE_CHECKING

import numpy as np
import pytest
from ase import Atoms

from motep.calculator import MTP
from motep.io.mlip.cfg import read_cfg
from motep.io.mlip.mtp import read_mtp

if TYPE_CHECKING:
    from motep.potentials.mtp.data import MTPData


def make_atoms(engine: str, level: int, data_path: pathlib.Path) -> Atoms:
    """Make the ASE `Atoms` object with the calculator."""
    path = data_path / f"fitting/crystals/multi/{level:02d}"
    if not (path / "pot.mtp").exists():
        pytest.skip()
    atoms = read_cfg(path / "out.cfg", index=-1)
    mtp_data_ref = read_mtp(path / "pot.mtp")
    atoms.calc = MTP(mtp_data=mtp_data_ref, engine=engine, is_trained=True)
    return atoms


@pytest.mark.parametrize("coeffs", ["moment_coeffs", "species_coeffs", "radial_coeffs"])
@pytest.mark.parametrize("level", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("engine", ["numpy", "numba"])
def test_jac_energy(
    engine: str,
    level: int,
    coeffs: str,
    data_path: pathlib.Path,
) -> None:
    """Test the Jacobian for the energy with respect to the parameters."""
    atoms = make_atoms(engine, level, data_path)

    atoms.get_potential_energy()
    jac_anl = atoms.calc.engine.jac_energy(atoms)

    dx = 1e-6

    jac_nmr = np.full_like(getattr(jac_anl, coeffs), np.nan)

    mtp_data: MTPData = copy.deepcopy(atoms.calc.engine.mtp_data)
    array = getattr(mtp_data, coeffs)

    for i in range(jac_nmr.size):
        orig = array.flat[i]

        array.flat[i] = orig + dx
        atoms.calc.update_parameters(mtp_data)
        ep = atoms.get_potential_energy()

        array.flat[i] = orig - dx
        atoms.calc.update_parameters(mtp_data)
        em = atoms.get_potential_energy()

        jac_nmr.flat[i] = (ep - em) / (2.0 * dx)

        array.flat[i] = orig

    print(jac_nmr)
    print(getattr(jac_anl, coeffs))

    np.testing.assert_allclose(jac_nmr, getattr(jac_anl, coeffs), atol=1e-4)


@pytest.mark.parametrize("coeffs", ["moment_coeffs", "species_coeffs", "radial_coeffs"])
@pytest.mark.parametrize("level", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("engine", ["numpy", "numba"])
def test_jac_forces(
    engine: str,
    level: int,
    coeffs: str,
    data_path: pathlib.Path,
) -> None:
    """Test the Jacobian for the forces with respect to the parameters."""
    atoms = make_atoms(engine, level, data_path)

    atoms.get_potential_energy()
    jac_anl = atoms.calc.engine.jac_forces(atoms)

    dx = 1e-6

    jac_nmr = np.full_like(getattr(jac_anl, coeffs), np.nan)

    mtp_data: MTPData = copy.deepcopy(atoms.calc.engine.mtp_data)
    array = getattr(mtp_data, coeffs)

    for indices, orig in np.ndenumerate(array):
        array[indices] = orig + dx
        atoms.calc.update_parameters(mtp_data)
        fp = atoms.get_forces()

        array[indices] = orig - dx
        atoms.calc.update_parameters(mtp_data)
        fm = atoms.get_forces()

        jac_nmr[indices] = (fp - fm) / (2.0 * dx)

        array[indices] = orig

    print(jac_nmr)
    print(getattr(jac_anl, coeffs))

    np.testing.assert_allclose(jac_nmr, getattr(jac_anl, coeffs), atol=1e-4)


@pytest.mark.parametrize("coeffs", ["moment_coeffs", "species_coeffs", "radial_coeffs"])
@pytest.mark.parametrize("level", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("engine", ["numpy", "numba"])
def test_jac_stress(
    engine: str,
    level: int,
    coeffs: str,
    data_path: pathlib.Path,
) -> None:
    """Test the Jacobian for the forces with respect to the parameters."""
    atoms = make_atoms(engine, level, data_path)

    atoms.get_potential_energy()
    jac_anl = atoms.calc.engine.jac_stress(atoms)

    dx = 1e-6

    jac_nmr = np.full_like(getattr(jac_anl, coeffs), np.nan)

    mtp_data: MTPData = copy.deepcopy(atoms.calc.engine.mtp_data)
    array = getattr(mtp_data, coeffs)

    for indices, orig in np.ndenumerate(array):
        array[indices] = orig + dx
        atoms.calc.update_parameters(mtp_data)
        sp = atoms.get_stress(voigt=False)

        array[indices] = orig - dx
        atoms.calc.update_parameters(mtp_data)
        sm = atoms.get_stress(voigt=False)

        jac_nmr[indices] = (sp - sm) / (2.0 * dx)

        array[indices] = orig

    print(jac_nmr)
    print(getattr(jac_anl, coeffs))

    np.testing.assert_allclose(jac_nmr, getattr(jac_anl, coeffs), atol=1e-4)
