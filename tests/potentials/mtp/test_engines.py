"""Tests for PyMTP."""

import pathlib

import numpy as np
import pytest

from motep.io.mlip.cfg import read_cfg
from motep.io.mlip.mtp import read_mtp
from motep.potentials.mtp.base import EngineBase
from motep.potentials.mtp.cext.engine import CExtMTPEngine
from motep.potentials.mtp.numpy.engine import NumpyMTPEngine

try:
    from motep.potentials.mtp.numba.engine import NumbaMTPEngine

    _numba_available = True
except ImportError:
    NumbaMTPEngine = None  # type: ignore[assignment,misc]
    _numba_available = False

try:
    from motep.potentials.mtp.jax.engine import JaxMTPEngine

    _jax_available = True
except ImportError:
    JaxMTPEngine = None  # type: ignore[assignment,misc]
    _jax_available = False

_numba_param = pytest.param(
    NumbaMTPEngine,
    marks=pytest.mark.skipif(not _numba_available, reason="numba not available"),
)
_jax_param = pytest.param(
    JaxMTPEngine,
    marks=pytest.mark.skipif(not _jax_available, reason="JAX not available"),
)


@pytest.mark.parametrize("level", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("molecule", [762, 291, 14214, 23208])
@pytest.mark.parametrize("mode", ["run", "train"])
@pytest.mark.parametrize(
    "engine", [NumpyMTPEngine, _numba_param, _jax_param, CExtMTPEngine]
)
# @pytest.mark.parametrize("molecule", [762])
def test_molecules(
    engine: type[EngineBase],
    mode: str,
    molecule: int,
    level: int,
    data_path: pathlib.Path,
) -> None:
    """Test PyMTP."""
    path = data_path / f"fitting/molecules/{molecule}/{level:02d}"
    if not (path / "pot.mtp").exists():
        pytest.skip("Test data not available")
    mtp_data = read_mtp(path / "pot.mtp")
    mtp = engine(mtp_data, mode=mode)
    images = [read_cfg(path / "out.cfg", index=0)]

    results_all = [mtp.calculate(atoms) for atoms in images]

    energies_ref = np.array([_.get_potential_energy() for _ in images])
    energies = np.array([_["energy"] for _ in results_all]).reshape(-1)
    np.testing.assert_allclose(energies, energies_ref)

    forces_ref = np.vstack([_.get_forces() for _ in images])
    forces = np.vstack([_["forces"] for _ in results_all])
    np.testing.assert_allclose(forces, forces_ref, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize("level", [2, 4, 6, 8, 10])
# @pytest.mark.parametrize("crystal", ["cubic", "noncubic"])
@pytest.mark.parametrize("crystal", ["multi"])
@pytest.mark.parametrize("mode", ["run", "train"])
# @pytest.mark.parametrize("engine", [NumpyMTPEngine, NumbaMTPEngine])
@pytest.mark.parametrize("engine", [_numba_param, _jax_param, CExtMTPEngine])
def test_crystals(
    engine: type[EngineBase],
    mode: str,
    crystal: int,
    level: int,
    data_path: pathlib.Path,
) -> None:
    """Test PyMTP."""
    path = data_path / f"fitting/crystals/{crystal}/{level:02d}"
    if not (path / "pot.mtp").exists():
        pytest.skip("Test data not available")
    mtp_data = read_mtp(path / "pot.mtp")
    mtp = engine(mtp_data, mode=mode)
    images = [read_cfg(path / "out.cfg", index=-1)]

    results_all = [mtp.calculate(atoms) for atoms in images]

    energies_ref = np.array([_.get_potential_energy() for _ in images])
    energies = np.array([_["energy"] for _ in results_all]).reshape(-1)
    np.testing.assert_allclose(energies, energies_ref)

    forces_ref = np.vstack([_.get_forces() for _ in images])
    forces = np.vstack([_["forces"] for _ in results_all])
    np.testing.assert_allclose(forces, forces_ref, rtol=0.0, atol=1e-6)

    stress_ref = np.vstack([_.get_stress() for _ in images])
    stress = np.vstack([_["stress"] for _ in results_all])
    np.testing.assert_allclose(stress, stress_ref, rtol=0.0, atol=1e-4)


# @pytest.mark.parametrize("level", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("level", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("molecule", [762, 291, 14214, 23028])
@pytest.mark.parametrize(
    "engine", [NumpyMTPEngine, _numba_param, _jax_param, CExtMTPEngine]
)
def test_forces(
    engine: type[EngineBase],
    molecule: int,
    level: int,
    data_path: pathlib.Path,
) -> None:
    """Test if forces are consistent with finite-difference values."""
    path = data_path / f"fitting/molecules/{molecule}/{level:02d}"
    if not (path / "pot.mtp").exists():
        pytest.skip()
    mtp_data = read_mtp(path / "pot.mtp")
    mtp = engine(mtp_data)
    atoms_ref = read_cfg(path / "out.cfg", index=-1)

    forces_ref = mtp.calculate(atoms_ref)["forces"]

    dx = 1e-6

    atoms = atoms_ref.copy()
    atoms.positions[0, 0] += dx
    ep = mtp.calculate(atoms)["energy"]

    atoms = atoms_ref.copy()
    atoms.positions[0, 0] -= dx
    em = mtp.calculate(atoms)["energy"]

    f = -1.0 * (ep - em) / (2.0 * dx)

    assert forces_ref[0, 0] == pytest.approx(f, abs=1e-4)


@pytest.mark.parametrize("level", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("crystal", ["cubic", "noncubic"])
@pytest.mark.parametrize(
    "engine",
    [NumpyMTPEngine, _numba_param, _jax_param, CExtMTPEngine],
)
def test_stress(
    engine: type[EngineBase],
    crystal: int,
    level: int,
    data_path: pathlib.Path,
) -> None:
    """Test if stresses are consistent with finite-difference values.

    Notes
    -----
    At this moment, the test fails for "noncubic" with index "-1" (distorted
    atomic positions). The reason is not yet clear and to be fixed later.

    """
    path = data_path / f"fitting/crystals/{crystal}/{level:02d}"
    if not (path / "pot.mtp").exists():
        pytest.skip()
    mtp_data = read_mtp(path / "pot.mtp")
    mtp = engine(mtp_data)
    atoms_ref = read_cfg(path / "out.cfg", index=-1)
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
        eplus = mtp.calculate(atoms)["energy"]

        x[i, i] = 1.0 - eps
        atoms = atoms_ref.copy()
        atoms.set_cell(cell @ x, scale_atoms=True)
        eminus = mtp.calculate(atoms)["energy"]

        stress[i, i] = (eplus - eminus) / (2.0 * eps * volume)
        x[i, i] = 1.0

        j = i - 2
        x[i, j] = x[j, i] = +0.5 * eps
        atoms = atoms_ref.copy()
        atoms.set_cell(cell @ x, scale_atoms=True)
        eplus = mtp.calculate(atoms)["energy"]

        x[i, j] = x[j, i] = -0.5 * eps
        atoms = atoms_ref.copy()
        atoms.set_cell(cell @ x, scale_atoms=True)
        eminus = mtp.calculate(atoms)["energy"]

        stress[i, j] = stress[j, i] = (eplus - eminus) / (2 * eps * volume)

    stress = stress.ravel()[[0, 4, 8, 5, 2, 1]]
    np.testing.assert_allclose(stress, stress_ref, rtol=0.0, atol=1e-4)


@pytest.mark.parametrize("level", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("crystal", ["multi"])
@pytest.mark.parametrize("mode", ["run", "train"])
@pytest.mark.parametrize("engine", [_numba_param, CExtMTPEngine])
def test_basis_data(
    engine: type[EngineBase],
    mode: str,
    crystal: int,
    level: int,
    data_path: pathlib.Path,
) -> None:
    """Test PyMTP."""
    path = data_path / f"fitting/crystals/{crystal}/{level:02d}"
    if not (path / "pot.mtp").exists():
        pytest.skip()
    mtp_data = read_mtp(path / "pot.mtp")
    # Assume NumpyMTPEngine as reference
    ref = NumpyMTPEngine(mtp_data, mode=mode)
    mtp = engine(mtp_data, mode=mode)
    images = [read_cfg(path / "out.cfg", index=-1)]

    for atoms in images:
        ref.calculate(atoms)
        mtp.calculate(atoms)

        mbd = mtp.mbd
        mbd_ref = ref.mbd
        np.testing.assert_allclose(mbd.vatoms, mbd_ref.vatoms, rtol=0.0, atol=1e-6)

        if mode == "train":
            np.testing.assert_allclose(mbd.dbdris, mbd_ref.dbdris, rtol=0.0, atol=1e-6)
            np.testing.assert_allclose(mbd.dbdeps, mbd_ref.dbdeps, rtol=0.0, atol=1e-6)
            np.testing.assert_allclose(mbd.dedcs, mbd_ref.dedcs, rtol=0.0, atol=1e-6)
            np.testing.assert_allclose(mbd.dgdcs, mbd_ref.dgdcs, rtol=0.0, atol=1e-6)
            np.testing.assert_allclose(mbd.dsdcs, mbd_ref.dsdcs, rtol=0.0, atol=1e-6)

            rbd = mtp.rbd
            rbd_ref = ref.rbd
            np.testing.assert_allclose(rbd.values, rbd_ref.values, rtol=0.0, atol=1e-6)
            np.testing.assert_allclose(rbd.dqdris, rbd_ref.dqdris, rtol=0.0, atol=1e-6)
            np.testing.assert_allclose(rbd.dqdeps, rbd_ref.dqdeps, rtol=0.0, atol=1e-6)
