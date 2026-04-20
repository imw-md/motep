"""Tests for Magnetic MTP (MMTP) implementations.

Tests to ensure that cext and numba implementations produce consistent results
for energies, forces, stress, and magnetic gradients.
"""

import pathlib

import numpy as np
import pytest

from motep.io.mlip.cfg import read_cfg
from motep.io.mlip.mtp import read_mmtp
from motep.potentials.mmtp.base import MagEngineBase
from motep.potentials.mmtp.cext.engine import CExtMagMTPEngine
from motep.potentials.mmtp.numba.engine import NumbaMagMTPEngine


@pytest.fixture
def mag_engine_and_atoms(data_path: pathlib.Path):
    path = data_path / "original/mag"
    mtp_path = path / "02.mmtp"
    if not mtp_path.exists():
        pytest.skip(f"Data file {mtp_path} not found")
    mtp_data = read_mmtp(mtp_path)
    engine = NumbaMagMTPEngine(mtp_data)
    atoms = read_cfg(path / "mag.cfg", index=-1)
    atoms.set_initial_magnetic_moments(atoms.get_magnetic_moments())
    return engine, atoms


class TestCollinearMagmoms:
    def test_2d_all_zero_columns_accepted(self, mag_engine_and_atoms) -> None:
        engine, atoms = mag_engine_and_atoms
        magmoms_2d = np.zeros((len(atoms), 3))
        result = engine.calculate(atoms, magmoms=magmoms_2d)
        assert "energy" in result

    def test_2d_noncollinear_raises(self, mag_engine_and_atoms) -> None:
        engine, atoms = mag_engine_and_atoms
        magmoms_2d = np.zeros((len(atoms), 3))
        magmoms_2d[:, 0] = 1.0
        magmoms_2d[:, 2] = 1.0
        with pytest.raises(ValueError, match="Non-collinear"):
            engine.calculate(atoms, magmoms=magmoms_2d)


@pytest.mark.parametrize("level", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("mode", ["run", "train", "train_mgrad"])
@pytest.mark.parametrize("engine_class", [CExtMagMTPEngine])
def test_mmtp_energies_forces_stress(
    level: int,
    mode: str,
    engine_class: type[MagEngineBase],
    data_path: pathlib.Path,
) -> None:
    """Test that cext and numba implementations produce identical energies, forces, and stress.

    This test loads a magnetic MTP model at the specified level and compares
    the results from the C extension and Numba implementations.
    """
    path = data_path / "original/mag"
    mtp_path = path / f"{level:02d}.mmtp"
    if not mtp_path.exists():
        pytest.skip(f"Data file {mtp_path} not found")

    mtp_data = read_mmtp(mtp_path)

    # Create engines with both implementations
    try:
        engine = engine_class(mtp_data, mode=mode)
    except NotImplementedError as e:
        pytest.skip(f"Engine does not support this configuration: {e}")

    try:
        ref_engine = NumbaMagMTPEngine(mtp_data, mode=mode)
    except NotImplementedError as e:
        pytest.skip(f"Reference engine does not support this configuration: {e}")

    atoms = read_cfg(path / "mag.cfg", index=-1)
    atoms.set_initial_magnetic_moments(atoms.get_magnetic_moments())

    # Calculate with both engines
    result = engine.calculate(atoms)
    ref_result = ref_engine.calculate(atoms)

    # Compare energies
    np.testing.assert_allclose(
        result["energy"],
        ref_result["energy"],
        rtol=1e-10,
        atol=1e-12,
        err_msg=f"Energies differ at level {level}, mode {mode}, engine {engine_class}",
    )

    # Compare forces
    np.testing.assert_allclose(
        result["forces"],
        ref_result["forces"],
        rtol=1e-10,
        atol=1e-12,
        err_msg=f"Forces differ at level {level}, mode {mode}, engine {engine_class}",
    )

    # Compare stress
    np.testing.assert_allclose(
        result["stress"],
        ref_result["stress"],
        rtol=1e-10,
        atol=1e-12,
        err_msg=f"Stress differs at level {level}, mode {mode}, engine {engine_class}",
    )

    # Compare magnetic gradients
    np.testing.assert_allclose(
        result["mgrad"],
        ref_result["mgrad"],
        rtol=1e-10,
        atol=1e-12,
        err_msg=f"Mgrad differs at level {level}, mode {mode}, engine {engine_class}",
    )


@pytest.mark.parametrize("level", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("mode", ["run", "train", "train_mgrad"])
@pytest.mark.parametrize("engine_class", [CExtMagMTPEngine, NumbaMagMTPEngine])
def test_mgrad(
    engine_class: type[MagEngineBase],
    level: int,
    mode: str,
    data_path: pathlib.Path,
) -> None:
    """Test if magnetic gradients are consistent with finite-difference values."""
    path = data_path / "original/mag"
    mtp_path = path / f"{level:02d}.mmtp"
    if not mtp_path.exists():
        pytest.skip(f"Data file {mtp_path} not found")

    mtp_data = read_mmtp(mtp_path)

    try:
        engine = engine_class(mtp_data, mode=mode)
    except NotImplementedError as e:
        pytest.skip(f"Engine does not support this configuration: {e}")

    atoms_ref = read_cfg(path / "mag.cfg", index=-1)
    atoms_ref.set_initial_magnetic_moments(atoms_ref.get_magnetic_moments())

    mag_grad_ref = engine.calculate(atoms_ref)["mgrad"]

    dx = 1e-6

    atoms = atoms_ref.copy()
    atoms.arrays["initial_magmoms"][0] += dx
    ep = engine.calculate(atoms)["energy"]

    atoms = atoms_ref.copy()
    atoms.arrays["initial_magmoms"][0] -= dx
    em = engine.calculate(atoms)["energy"]

    t = +1.0 * (ep - em) / (2.0 * dx)

    assert mag_grad_ref[0] == pytest.approx(t, abs=1e-4)


@pytest.mark.parametrize("level", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("engine_class", [CExtMagMTPEngine, NumbaMagMTPEngine])
def test_forces(
    engine_class: type[MagEngineBase],
    level: int,
    data_path: pathlib.Path,
) -> None:
    """Test if MMTP forces are consistent with finite-difference values."""
    path = data_path / "original/mag"
    mtp_path = path / f"{level:02d}.mmtp"
    if not mtp_path.exists():
        pytest.skip(f"Data file {mtp_path} not found")

    mtp_data = read_mmtp(mtp_path)

    try:
        engine = engine_class(mtp_data)
    except NotImplementedError as e:
        pytest.skip(f"Engine does not support this configuration: {e}")

    atoms_ref = read_cfg(path / "mag.cfg", index=-1)
    atoms_ref.set_initial_magnetic_moments(atoms_ref.get_magnetic_moments())

    forces_ref = engine.calculate(atoms_ref)["forces"]

    dx = 1e-6

    atoms = atoms_ref.copy()
    atoms.positions[0, 0] += dx
    ep = engine.calculate(atoms)["energy"]

    atoms = atoms_ref.copy()
    atoms.positions[0, 0] -= dx
    em = engine.calculate(atoms)["energy"]

    f = -1.0 * (ep - em) / (2.0 * dx)

    assert forces_ref[0, 0] == pytest.approx(f, abs=1e-4)


@pytest.mark.parametrize("level", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("engine_class", [CExtMagMTPEngine, NumbaMagMTPEngine])
def test_stress(
    engine_class: type[MagEngineBase],
    level: int,
    data_path: pathlib.Path,
) -> None:
    """Test if MMTP stresses are consistent with finite-difference values."""
    path = data_path / "original/mag"
    mtp_path = path / f"{level:02d}.mmtp"
    if not mtp_path.exists():
        pytest.skip(f"Data file {mtp_path} not found")

    mtp_data = read_mmtp(mtp_path)

    try:
        mtp = engine_class(mtp_data)
    except NotImplementedError as e:
        pytest.skip(f"Engine does not support this configuration: {e}")

    atoms_ref = read_cfg(path / "mag.cfg", index=-1)
    magmoms = atoms_ref.get_magnetic_moments().copy()
    atoms_ref.set_initial_magnetic_moments(magmoms)

    stress_ref = mtp.calculate(atoms_ref)["stress"]

    stress = np.zeros((3, 3), dtype=float)
    eps = 1e-6
    cell = atoms_ref.cell.copy()
    volume = atoms_ref.get_volume()
    for i in range(3):
        x = np.eye(3)
        x[i, i] = 1.0 + eps
        atoms = atoms_ref.copy()
        atoms.set_cell(cell @ x, scale_atoms=True)
        atoms.set_initial_magnetic_moments(magmoms)
        eplus = mtp.calculate(atoms)["energy"]

        x[i, i] = 1.0 - eps
        atoms = atoms_ref.copy()
        atoms.set_cell(cell @ x, scale_atoms=True)
        atoms.set_initial_magnetic_moments(magmoms)
        eminus = mtp.calculate(atoms)["energy"]

        stress[i, i] = (eplus - eminus) / (2.0 * eps * volume)
        x[i, i] = 1.0

        j = i - 2
        x[i, j] = x[j, i] = +0.5 * eps
        atoms = atoms_ref.copy()
        atoms.set_cell(cell @ x, scale_atoms=True)
        atoms.set_initial_magnetic_moments(magmoms)
        eplus = mtp.calculate(atoms)["energy"]

        x[i, j] = x[j, i] = -0.5 * eps
        atoms = atoms_ref.copy()
        atoms.set_cell(cell @ x, scale_atoms=True)
        atoms.set_initial_magnetic_moments(magmoms)
        eminus = mtp.calculate(atoms)["energy"]

        stress[i, j] = stress[j, i] = (eplus - eminus) / (2 * eps * volume)

    stress = stress.ravel()[[0, 4, 8, 5, 2, 1]]
    np.testing.assert_allclose(stress, stress_ref, rtol=0.0, atol=1e-4)
