"""Tests for PyMTP."""

import pathlib
from typing import Any

import numpy as np
import pytest

from motep.io.mlip.cfg import read_cfg
from motep.io.mlip.mtp import read_mtp
from motep.potentials.mtp.jax.engine import JaxMTPEngine
from motep.potentials.mtp.numba.engine import NumbaMTPEngine
from motep.potentials.mtp.numpy.engine import NumpyMTPEngine


def get_scale(component: str, d: float) -> np.ndarray:
    """Get the scaling matrix for the corresponding stress component."""
    voigt_index = ["xx", "yy", "zz", "yz", "zx", "xy"].index(component)
    if component == "xx":
        scale = np.diag((1.0 + d, 1.0, 1.0))
    elif component == "yy":
        scale = np.diag((1.0, 1.0 + d, 1.0))
    elif component == "zz":
        scale = np.diag((1.0, 1.0, 1.0 + d))
    elif component == "yz":
        scale = np.array(((1.0, 0.0, 0.0), (0.0, 1.0, 0.5 * d), (0.0, 0.5 * d, 1.0)))
    elif component == "zx":
        scale = np.array(((1.0, 0.0, 0.5 * d), (0.0, 1.0, 0.0), (0.5 * d, 0.0, 1.0)))
    elif component == "xy":
        scale = np.array(((1.0, 0.5 * d, 0.0), (0.5 * d, 1.0, 0.0), (0.0, 0.0, 1.0)))
    else:
        raise ValueError(component)
    return voigt_index, scale


@pytest.mark.parametrize("level", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("molecule", [762, 291, 14214, 23208])
@pytest.mark.parametrize("is_trained", [False, True])
@pytest.mark.parametrize("engine", [NumpyMTPEngine, NumbaMTPEngine, JaxMTPEngine])
# @pytest.mark.parametrize("molecule", [762])
def test_molecules(
    engine: Any,
    is_trained: bool,
    molecule: int,
    level: int,
    data_path: pathlib.Path,
) -> None:
    """Test PyMTP."""
    path = data_path / f"fitting/molecules/{molecule}/{level:02d}"
    if not (path / "pot.mtp").exists():
        pytest.skip()
    parameters = read_mtp(path / "pot.mtp")
    # parameters["species"] = species
    mtp = engine(parameters, is_trained=is_trained)
    images = [read_cfg(path / "out.cfg", index=0)]
    mtp._initiate_neighbor_list(images[0])

    results_all = [mtp.calculate(atoms) for atoms in images]

    energies_ref = np.array([_.get_potential_energy() for _ in images])
    energies = np.array([_["energy"] for _ in results_all]).reshape(-1)
    print(np.array(energies), np.array(energies_ref))
    np.testing.assert_allclose(energies, energies_ref)

    forces_ref = np.vstack([_.get_forces() for _ in images])
    forces = np.vstack([_["forces"] for _ in results_all])
    print(np.array(forces), np.array(forces_ref))
    np.testing.assert_allclose(forces, forces_ref, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize("level", [2, 4, 6, 8, 10])
# @pytest.mark.parametrize("crystal", ["cubic", "noncubic"])
@pytest.mark.parametrize("crystal", ["size", "multi"])
@pytest.mark.parametrize("is_trained", [False, True])
# @pytest.mark.parametrize("engine", [NumpyMTPEngine, NumbaMTPEngine])
@pytest.mark.parametrize("engine", [NumbaMTPEngine, JaxMTPEngine])
def test_crystals(
    engine: Any,
    is_trained: bool,
    crystal: int,
    level: int,
    data_path: pathlib.Path,
) -> None:
    """Test PyMTP."""
    path = data_path / f"fitting/crystals/{crystal}/{level:02d}"
    if not (path / "pot.mtp").exists():
        pytest.skip()
    parameters = read_mtp(path / "pot.mtp")
    # parameters["species"] = species
    mtp = engine(parameters, is_trained=is_trained)
    images = [read_cfg(path / "out.cfg", index=-1)]
    mtp._initiate_neighbor_list(images[0])

    results_all = [mtp.calculate(atoms) for atoms in images]

    energies_ref = np.array([_.get_potential_energy() for _ in images])
    energies = np.array([_["energy"] for _ in results_all]).reshape(-1)
    print(np.array(energies), np.array(energies_ref))
    np.testing.assert_allclose(energies, energies_ref)

    forces_ref = np.vstack([_.get_forces() for _ in images])
    forces = np.vstack([_["forces"] for _ in results_all])
    print(np.array(forces), np.array(forces_ref))
    np.testing.assert_allclose(forces, forces_ref, rtol=0.0, atol=1e-6)

    stress_ref = np.vstack([_.get_stress() for _ in images])
    stress = np.vstack([_["stress"] for _ in results_all])
    print(np.array(stress), np.array(stress_ref))
    np.testing.assert_allclose(stress, stress_ref, rtol=0.0, atol=1e-4)


# @pytest.mark.parametrize("level", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("level", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("molecule", [762, 291, 14214, 23028])
@pytest.mark.parametrize("engine", [NumpyMTPEngine, NumbaMTPEngine, JaxMTPEngine])
def test_forces(
    engine: Any,
    molecule: int,
    level: int,
    data_path: pathlib.Path,
) -> None:
    """Test if forces are consistent with finite-difference values."""
    path = data_path / f"fitting/molecules/{molecule}/{level:02d}"
    if not (path / "pot.mtp").exists():
        pytest.skip()
    parameters = read_mtp(path / "pot.mtp")
    # parameters["species"] = species
    mtp = engine(parameters)
    atoms_ref = read_cfg(path / "out.cfg", index=-1)
    mtp._initiate_neighbor_list(atoms_ref)

    forces_ref = mtp.calculate(atoms_ref)["forces"]

    dx = 1e-6

    atoms = atoms_ref.copy()
    atoms.positions[0, 0] += dx
    ep = mtp.calculate(atoms)["energy"]

    atoms = atoms_ref.copy()
    atoms.positions[0, 0] -= dx
    em = mtp.calculate(atoms)["energy"]

    f = -1.0 * (ep - em) / (2.0 * dx)

    print(forces_ref[0, 0], f)

    assert forces_ref[0, 0] == pytest.approx(f, abs=1e-4)


@pytest.mark.parametrize("component", ["xx", "yy", "zz", "yz", "zx", "xy"])
@pytest.mark.parametrize("level", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("crystal", ["cubic", "noncubic"])
@pytest.mark.parametrize("engine", [NumpyMTPEngine, NumbaMTPEngine, JaxMTPEngine])
def test_stress(
    engine: Any,
    crystal: int,
    level: int,
    component: str,
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
    parameters = read_mtp(path / "pot.mtp")
    # parameters["species"] = species
    mtp = engine(parameters)
    atoms_ref = read_cfg(path / "out.cfg", index=-1)
    mtp._initiate_neighbor_list(atoms_ref)

    stress_ref = mtp.calculate(atoms_ref)["stress"]

    dx = 1e-6

    atoms = atoms_ref.copy()
    sindex, scale = get_scale(component, +1.0 * dx)
    atoms.set_cell(atoms.get_cell() @ scale, scale_atoms=True)
    ep = mtp.calculate(atoms)["energy"]

    atoms = atoms_ref.copy()
    sindex, scale = get_scale(component, -1.0 * dx)
    atoms.set_cell(atoms.get_cell() @ scale, scale_atoms=True)
    em = mtp.calculate(atoms)["energy"]

    s = (ep - em) / (2.0 * dx) / atoms_ref.get_volume()

    print(stress_ref[sindex], s)

    assert stress_ref[sindex] == pytest.approx(s, abs=1e-4)


@pytest.mark.parametrize("level", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("crystal", ["size", "multi"])
@pytest.mark.parametrize("engine", [NumbaMTPEngine])  # , JaxMTPEngine])
def test_basis_data(
    engine: Any,
    crystal: int,
    level: int,
    data_path: pathlib.Path,
) -> None:
    """Test PyMTP."""
    is_trained = True
    path = data_path / f"fitting/crystals/{crystal}/{level:02d}"
    if not (path / "pot.mtp").exists():
        pytest.skip()
    parameters = read_mtp(path / "pot.mtp")
    # Assume NumpyMTPEngine as reference
    ref = NumpyMTPEngine(parameters, is_trained=is_trained)
    mtp = engine(parameters, is_trained=is_trained)
    images = [read_cfg(path / "out.cfg", index=-1)]
    ref._initiate_neighbor_list(images[0])
    mtp._initiate_neighbor_list(images[0])

    for atoms in images:
        ref.calculate(atoms)
        mtp.calculate(atoms)

        mbd = mtp.mbd
        mbd_ref = ref.mbd
        np.testing.assert_allclose(mbd.values, mbd_ref.values, rtol=0.0, atol=1e-6)
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
