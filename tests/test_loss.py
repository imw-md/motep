"""Tests for trainer.py."""

import logging
import pathlib

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from motep.io.mlip.cfg import read_cfg
from motep.io.mlip.mtp import read_mtp
from motep.loss import ErrorPrinter, LossFunction, LossFunctionStress
from motep.parallel import world
from motep.setting import LossSetting

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("forces_per_atom", [False, True])
def test_without_forces(*, forces_per_atom: bool, data_path: pathlib.Path) -> None:
    """Test if `LossFunction` works for the training data without forces."""
    engine = "cext"
    level = 2
    path = data_path / f"fitting/crystals/multi/{level:02d}"
    if not (path / "pot.mtp").exists():
        pytest.skip()
    mtp_data = read_mtp(path / "pot.mtp")
    images = read_cfg(path / "out.cfg", index=":")[::1000]
    for atoms in images:
        del atoms.calc.results["forces"]

    setting = LossSetting(
        energy_weight=1.0,
        forces_weight=0.01,
        stress_weight=0.001,
        forces_per_atom=forces_per_atom,
    )

    loss = LossFunction(
        images,
        mtp_data=mtp_data,
        setting=setting,
        comm=world,
        engine=engine,
    )

    loss(mtp_data.parameters)
    loss.jac(mtp_data.parameters)
    ErrorPrinter(loss.images).log(logger)


@pytest.mark.parametrize(
    (
        "energy_per_atom",
        "forces_per_atom",
        "stress_times_volume",
        "energy_per_conf",
        "forces_per_conf",
        "stress_per_conf",
    ),
    [
        (True, True, False, True, True, True),  # default
        (False, True, False, True, True, True),
        (True, False, False, True, True, True),
        (True, True, True, True, True, True),
        (True, True, False, False, True, True),
        (True, True, False, True, False, True),
        (True, True, False, True, True, False),
    ],
)
@pytest.mark.parametrize("level", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("engine", ["cext"])
def test_jac(
    *,
    engine: str,
    level: int,
    energy_per_atom: bool,
    forces_per_atom: bool,
    stress_times_volume: bool,
    energy_per_conf: bool,
    forces_per_conf: bool,
    stress_per_conf: bool,
    data_path: pathlib.Path,
) -> None:
    """Test the Jacobian for the forces with respect to the parameters."""
    path = data_path / f"fitting/crystals/multi/{level:02d}"
    if not (path / "pot.mtp").exists():
        pytest.skip()
    mtp_data = read_mtp(path / "pot.mtp")
    images = [read_cfg(path / "out.cfg", index=-1)]

    setting = LossSetting(
        energy_weight=1.0,
        forces_weight=1.0 if forces_per_atom else 0.01,
        stress_weight=0.001,
        energy_per_atom=energy_per_atom,
        forces_per_atom=forces_per_atom,
        stress_times_volume=stress_times_volume,
        energy_per_conf=energy_per_conf,
        forces_per_conf=forces_per_conf,
        stress_per_conf=stress_per_conf,
    )

    loss = LossFunction(
        images,
        mtp_data=mtp_data,
        setting=setting,
        comm=world,
        engine=engine,
    )
    loss(mtp_data.parameters)
    jac_anl = loss.jac(mtp_data.parameters)

    jac_nmr = np.full_like(jac_anl, np.nan)

    dx = 1e-9

    parameters = mtp_data.parameters

    for i, orig in enumerate(parameters):
        parameters[i] = orig + dx
        lp = loss(parameters)

        parameters[i] = orig - dx
        lm = loss(parameters)

        jac_nmr[i] = (lp - lm) / (2.0 * dx)

        parameters[i] = orig

    print(jac_nmr)
    print(jac_anl)

    assert np.any(jac_nmr)  # check if some of the elements are non-zero

    np.testing.assert_allclose(jac_nmr, jac_anl, rtol=2e-1, atol=1e-12)


def test_stress_weight_scaling() -> None:
    """Test that the stress weight scales as intended.

    With stress_times_volume=True and energy_per_atom=False the per-configuration
    weight factor is V^2.  Enabling energy_per_atom=True additionally multiplies
    by (1/N)^2, so the ratio between the two losses must equal 1/N^2.
    """
    n_atoms = 4
    a = 5.0
    atoms = Atoms(
        "H" * n_atoms,
        positions=[(i * a / n_atoms, 0.0, 0.0) for i in range(n_atoms)],
        cell=[a, a, a],
        pbc=True,
    )

    # Nonzero residual so the loss is nonzero
    stress_result = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
    stress_target = np.zeros(6)

    calc = SinglePointCalculator(atoms)
    calc.results["stress"] = stress_result
    calc.targets = {"stress": stress_target}
    atoms.calc = calc

    images = [atoms]

    # stress_times_volume=True, energy_per_atom=False -> weight = V^2
    loss_no_epa = LossFunctionStress(
        images,
        mtp_data=None,
        stress_times_volume=True,
        energy_per_atom=False,
        comm=world,
    )
    val_no_epa = loss_no_epa.calculate()

    # stress_times_volume=True, energy_per_atom=True -> weight = V^2 / N^2
    loss_with_epa = LossFunctionStress(
        images,
        mtp_data=None,
        stress_times_volume=True,
        energy_per_atom=True,
        comm=world,
    )
    val_with_epa = loss_with_epa.calculate()

    expected_ratio = 1.0 / n_atoms**2
    np.testing.assert_allclose(val_with_epa / val_no_epa, expected_ratio)

    loss_no_stv = LossFunctionStress(
        images,
        mtp_data=None,
        stress_times_volume=False,
        comm=world,
    )
    val_no_stv = loss_no_stv.calculate()
    volume = a**3
    expected_ratio = 1.0 / volume**2
    np.testing.assert_allclose(val_no_stv / val_no_epa, expected_ratio)
