"""Engine-level Jacobian tests for MMTP (magnetic MTP)."""

import copy
import pathlib
from typing import TYPE_CHECKING

import numpy as np
import pytest

from motep.calculator import MMTP
from motep.io.mlip.cfg import read_cfg
from motep.io.mlip.mtp import read_mmtp

if TYPE_CHECKING:
    from motep.potentials.mmtp.data import MagMTPData


def make_mag_atoms(
    engine: str,
    level: int,
    data_path: pathlib.Path,
) -> "MMTP":
    """Make ASE Atoms with an MMTP calculator in train mode."""
    path = data_path / "original/mag"
    mtp_path = path / f"{level:02d}.mmtp"
    if not mtp_path.exists():
        pytest.skip(f"Data file {mtp_path} not found")
    mtp_data = read_mmtp(mtp_path)
    atoms = read_cfg(path / "mag.cfg", index=-1)
    atoms.set_initial_magnetic_moments(atoms.get_magnetic_moments())
    atoms.calc = MMTP(mtp_data, engine=engine, mode="train", relax_magmoms=False)
    return atoms


@pytest.mark.parametrize("coeffs", ["moment_coeffs", "species_coeffs", "radial_coeffs"])
@pytest.mark.parametrize("level", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("engine", ["cext"])
def test_jac_energy(
    engine: str,
    level: int,
    coeffs: str,
    data_path: pathlib.Path,
) -> None:
    """Test analytical energy Jacobian against finite differences for MMTP."""
    atoms = make_mag_atoms(engine, level, data_path)

    atoms.get_potential_energy()
    jac_anl = atoms.calc.engine.jac_energy(atoms)

    dx = 1e-6

    # jac_anl.species_coeffs is raveled → always 1-D for energy
    jac_anl_coeffs = getattr(jac_anl, coeffs)
    jac_nmr = np.full_like(jac_anl_coeffs, np.nan)

    mtp_data: MagMTPData = copy.deepcopy(atoms.calc.engine.mtp_data)
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

    np.testing.assert_allclose(jac_nmr, jac_anl_coeffs, atol=1e-4)


@pytest.mark.parametrize("coeffs", ["moment_coeffs", "radial_coeffs"])
@pytest.mark.parametrize("level", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("engine", ["cext"])
def test_jac_forces(
    engine: str,
    level: int,
    coeffs: str,
    data_path: pathlib.Path,
) -> None:
    """Test analytical force Jacobian against finite differences for MMTP.

    Only moment_coeffs and radial_coeffs are tested; species_coeffs produces a
    zero Jacobian for forces in the MMTP formulation.
    """
    atoms = make_mag_atoms(engine, level, data_path)

    atoms.get_potential_energy()
    jac_anl = atoms.calc.engine.jac_forces(atoms)

    dx = 1e-6

    jac_anl_coeffs = getattr(jac_anl, coeffs)
    jac_nmr = np.full_like(jac_anl_coeffs, np.nan)

    mtp_data: MagMTPData = copy.deepcopy(atoms.calc.engine.mtp_data)
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

    np.testing.assert_allclose(jac_nmr, jac_anl_coeffs, atol=1e-4)
