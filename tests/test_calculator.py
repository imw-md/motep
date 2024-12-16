"""Tests for the ASE calculator."""

import pathlib
import shutil

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


@pytest.mark.parametrize("engine", ["numpy", "numba", "jax"])
def test_potential_energies(engine: str, data_path: pathlib.Path) -> None:
    """Test if the site-energies are computed."""
    molecule = 291
    level = 10
    path = data_path / f"fitting/molecules/{molecule}/{level:02d}"
    mtp_parameters = read_mtp(path / "pot.mtp")
    atoms = read_cfg(path / "out.cfg", index=0)
    atoms.calc = MTP(mtp_parameters, engine=engine)
    assert atoms.get_potential_energies().sum() == atoms.get_potential_energy()


def _modify_min_dist(path: pathlib.Path, min_dist: float) -> None:
    with path.open("r", encoding="utf-8") as fd:
        oldlines = fd.readlines()
    newlines = []
    for line in oldlines:
        if "min_dist" in line:
            newlines.append(f"\tmin_dist = {min_dist}\n")
        else:
            newlines.append(line)
    with path.open("w", encoding="utf-8") as fd:
        fd.write("".join(newlines))


def test_min_dist(data_path: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """Test if the treatment within `min_dist` is consistent with `mlippy`."""
    mlippy = pytest.importorskip("mlippy")
    molecule = 291
    level = 10
    path = data_path / f"fitting/molecules/{molecule}/{level:02d}"

    shutil.copy2(path / "pot.mtp", tmp_path / "pot.mtp")

    # larger than the C-H bond length
    _modify_min_dist(tmp_path / "pot.mtp", 1.5)

    atoms = read_cfg(path / "out.cfg", index=0)

    atoms.calc = MTP(read_mtp(tmp_path / "pot.mtp"), engine="numpy")
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    pot = mlippy.mtp(str(tmp_path / "pot.mtp"))
    for _ in range(2):
        pot.add_atomic_type(_)

    atoms.calc = mlippy.MLIP_Calculator(pot, {})
    energy_ref = atoms.get_potential_energy()
    forces_ref = atoms.get_forces()

    np.testing.assert_allclose(energy, energy_ref)
    np.testing.assert_allclose(forces, forces_ref, rtol=0.0, atol=1e-6)
