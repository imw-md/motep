"""Tests for the ASE calculator."""

import pathlib

import numpy as np
import pytest

from motep.calculator import MTP
from motep.io.mlip.cfg import read_cfg
from motep.io.mlip.mtp import read_mtp


def test_calculator(data_path: pathlib.Path) -> None:
    """Test the ASE calculator."""
    molecule = 291
    level = 10
    path = data_path / f"fitting/molecules/{molecule}/{level:02d}"
    mtp_parameters = read_mtp(path / "pot.mtp")
    calc = MTP(mtp_parameters, engine="numpy")
    atoms = read_cfg(path / "out.cfg", index=0)
    energy_ref = atoms.get_potential_energy()
    forces_ref = atoms.get_forces()
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    np.testing.assert_allclose(energy, energy_ref)
    np.testing.assert_allclose(forces, forces_ref, rtol=0.0, atol=1e-6)
    with pytest.raises(NotImplementedError):
        atoms.get_stress()
