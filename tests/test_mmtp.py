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


@pytest.mark.parametrize("level", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("mode", ["run", "train", "train_mgrad"])
@pytest.mark.parametrize("engine", [CExtMagMTPEngine])
def test_mmtp_energies_forces_stress(
    level: int,
    mode: str,
    engine: type[MagEngineBase],
    data_path: pathlib.Path,
) -> None:
    """Test that cext and numba implementations produce identical energies, forces, and stress.

    This test loads a magnetic MTP model at the specified level and compares
    the results from the C extension and Numba implementations. Skips if an engine
    does not support the basis type in the test file.
    """
    path = data_path / "original/mag"
    mtp_path = path / f"{level:02d}.mmtp"
    if not (mtp_path).exists():
        pytest.skip(f"Data file {mtp_path} not found")

    mtp_data = read_mmtp(mtp_path)

    # Create engines with both implementations
    try:
        engine_instance = engine(mtp_data, mode=mode)
    except NotImplementedError as e:
        pytest.skip(f"Engine does not support this configuration: {e}")

    try:
        ref_engine = NumbaMagMTPEngine(mtp_data, mode=mode)
    except NotImplementedError as e:
        pytest.skip(f"Reference engine does not support this configuration: {e}")

    atoms = read_cfg(path / "mag.cfg", index=-1)
    atoms.set_initial_magnetic_moments(atoms.get_magnetic_moments())

    # Calculate with both engines
    result = engine_instance.calculate(atoms)
    ref_result = ref_engine.calculate(atoms)

    # Compare energies
    np.testing.assert_allclose(
        result["energy"],
        ref_result["energy"],
        rtol=1e-10,
        atol=1e-12,
        err_msg=f"Energies differ at level {level}, mode {mode}, engine {engine}",
    )

    # Compare forces
    np.testing.assert_allclose(
        result["forces"],
        ref_result["forces"],
        rtol=1e-10,
        atol=1e-12,
        err_msg=f"Forces differ at level {level}, mode {mode}, engine {engine}",
    )

    # Compare stress
    np.testing.assert_allclose(
        result["stress"],
        ref_result["stress"],
        rtol=1e-10,
        atol=1e-12,
        err_msg=f"Stress differs at level {level}, mode {mode}, engine {engine}",
    )

    # Compare magnetic gradients
    np.testing.assert_allclose(
        result["mgrad"],
        ref_result["mgrad"],
        rtol=1e-10,
        atol=1e-12,
        err_msg=f"Mgrad differs at level {level}, mode {mode}, engine {engine}",
    )


@pytest.mark.parametrize("level", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("mode", ["run", "train", "train_mgrad"])
def test_mgrad(
    level: int,
    mode: str,
    data_path: pathlib.Path,
) -> None:
    """Test if magnetic gradients are consistent with finite-difference values.

    Uses only numba engine for testing, since energies and forces are compared
    in test_mmtp_energies_forces_stress. Skips if the engine does not support
    the basis type in the test file.
    """
    path = data_path / "original/mag"
    mtp_path = path / f"{level:02d}.mmtp"
    if not (mtp_path).exists():
        pytest.skip(f"Data file {mtp_path} not found")

    mtp_data = read_mmtp(mtp_path)

    try:
        mtp = NumbaMagMTPEngine(mtp_data, mode=mode)
    except NotImplementedError as e:
        pytest.skip(f"Engine does not support this configuration: {e}")

    atoms_ref = read_cfg(path / "mag.cfg", index=-1)
    atoms_ref.set_initial_magnetic_moments(atoms_ref.get_magnetic_moments())

    mag_grad_ref = mtp.calculate(atoms_ref)["mgrad"]

    dx = 1e-6

    atoms = atoms_ref.copy()
    atoms.arrays["initial_magmoms"][0] += dx
    ep = mtp.calculate(atoms)["energy"]

    atoms = atoms_ref.copy()
    atoms.arrays["initial_magmoms"][0] -= dx
    em = mtp.calculate(atoms)["energy"]

    t = +1.0 * (ep - em) / (2.0 * dx)

    assert mag_grad_ref[0] == pytest.approx(t, abs=1e-4)
