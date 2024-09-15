"""Tests for PyMTP."""

import pathlib

import numpy as np
import pytest

from motep.io.mlip.cfg import read_cfg
from motep.io.mlip.mtp import read_mtp
from motep.mtp import MTP, init_radial_basis_functions, calc_radial_basis


@pytest.mark.parametrize("level", [2, 4, 6, 8, 10])
@pytest.mark.parametrize(
    ("molecule", "species"),
    [(762, {1: 0}), (291, {6: 0, 1: 1}), (14214, {9: 0, 1: 1}), (23208, {8: 0})],
)
# @pytest.mark.parametrize("molecule", [762])
def test_molecules(
    molecule: int,
    species: dict[int, int],
    level: int,
    data_path: pathlib.Path,
) -> None:
    """Test PyMTP."""
    path = data_path / f"fitting/molecules/{molecule}/{level:02d}"
    if not (path / "pot.mtp").exists():
        pytest.skip()
    parameters = read_mtp(path / "pot.mtp")
    # parameters["species"] = species
    mtp = MTP(parameters)
    images = [read_cfg(path / "out.cfg", index=0)]
    mtp._initiate_neighbor_list(images[0])

    energies_ref = np.array([_.get_potential_energy() for _ in images])
    energies = np.array([mtp.get_energy(_)[0] for _ in images]).reshape(-1)
    print(np.array(energies), np.array(energies_ref))
    np.testing.assert_allclose(energies, energies_ref)

    forces_ref = np.vstack([_.get_forces() for _ in images])
    forces = np.vstack([mtp.get_energy(_)[1] for _ in images])
    print(np.array(forces), np.array(forces_ref))
    np.testing.assert_allclose(forces, forces_ref, rtol=0.0, atol=1e-6)


# @pytest.mark.parametrize("level", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("level", [2, 4, 6, 8, 10])
@pytest.mark.parametrize(
    "molecule, species",
    [(762, {1: 0}), (291, {6: 0, 1: 1}), (14214, {9: 0, 1: 1}), (23208, {8: 0})],
)
def test_forces(
    molecule: int,
    species: dict[int, int],
    level: int,
    data_path: pathlib.Path,
) -> None:
    """Test if forces are consistent with finite-difference values."""
    path = data_path / f"fitting/molecules/{molecule}/{level:02d}"
    if not (path / "pot.mtp").exists():
        pytest.skip()
    parameters = read_mtp(path / "pot.mtp")
    # parameters["species"] = species
    mtp = MTP(parameters)
    atoms_ref = read_cfg(path / "out.cfg", index=-1)
    mtp._initiate_neighbor_list(atoms_ref)

    forces_ref = mtp.get_energy(atoms_ref)[1]

    dx = 1e-6

    atoms = atoms_ref.copy()
    atoms.positions[0, 0] += dx
    ep = mtp.get_energy(atoms)[0]

    atoms = atoms_ref.copy()
    atoms.positions[0, 0] -= dx
    em = mtp.get_energy(atoms)[0]

    f = -1.0 * (ep - em) / (2.0 * dx)

    print(forces_ref[0, 0], f)

    assert forces_ref[0, 0] == pytest.approx(f, abs=1e-4)


params = [
    (
        np.array(
            [
                [
                    [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]],
                ],
            ]
        ),
        np.array([[2.0, 2.0, 1.0]]),
        [0],
    ),
    (
        np.array(
            [
                [
                    [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]],
                    [[2.0, 3.0, 4.0], [3.0, 4.0, 5.0]],
                ],
                [
                    [[4.0, 5.0, 6.0], [5.0, 6.0, 7.0]],
                    [[6.0, 7.0, 8.0], [7.0, 8.0, 9.0]],
                ],
            ]
        ),
        np.array([[2.0, 2.0, 1.0], [1.0, 1.0, 2.0]]),
        [0, 1],
    ),
]


@pytest.mark.parametrize(("radial_coeffs", "r_ijs", "jtypes"), params)
def test_radial_funcs(
    radial_coeffs: np.ndarray,
    r_ijs: np.ndarray,
    jtypes: list[int],
):
    radial_funcs_count = radial_coeffs.shape[2]
    radial_basis_functions = init_radial_basis_functions(
        radial_coeffs,
        min_dist=2.0,
        max_dist=5.0,
    )
    r_abs = np.linalg.norm(r_ijs, axis=1)
    _, radial_basis_derivs = calc_radial_basis(
        radial_basis_functions,
        r_abs,
        0,
        jtypes=jtypes,
        scaling=1.0,
        min_dist=2.0,
        max_dist=5.0,
        radial_funcs_count=radial_funcs_count,
    )
    radial_basis_derivs = radial_basis_derivs[:, :, None] * r_ijs / r_abs[:, None]
    dx = 1e-6
    r_ijs_p = r_ijs + [[dx, 0.0, 0.0]]
    r_abs_p = np.linalg.norm(r_ijs_p, axis=1)
    radial_basis_values_p, _ = calc_radial_basis(
        radial_basis_functions,
        r_abs_p,
        0,
        jtypes=jtypes,
        scaling=1.0,
        min_dist=2.0,
        max_dist=5.0,
        radial_funcs_count=radial_funcs_count,
    )
    r_ijs_m = r_ijs - [[dx, 0.0, 0.0]]
    r_abs_m = np.linalg.norm(r_ijs_m, axis=1)
    radial_basis_values_m, _ = calc_radial_basis(
        radial_basis_functions,
        r_abs_m,
        0,
        jtypes=jtypes,
        scaling=1.0,
        min_dist=2.0,
        max_dist=5.0,
        radial_funcs_count=radial_funcs_count,
    )
    radial_basis_derivs_fd = radial_basis_values_p - radial_basis_values_m
    radial_basis_derivs_fd /= 2.0 * dx
    np.testing.assert_allclose(
        radial_basis_derivs[0, 0, 0],
        radial_basis_derivs_fd[0, 0],
    )
