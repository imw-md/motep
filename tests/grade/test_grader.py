"""Tests for active-learning algorithms."""

from pathlib import Path

import numpy as np
import pytest

from motep.grade.grader import GradeMode, Grader
from motep.io.mlip.cfg import read_cfg
from motep.io.mlip.mtp import read_mtp

_optimized = [
    ["moment_coeffs"],
    ["moment_coeffs", "species_coeffs"],
    ["moment_coeffs", "species_coeffs", "radial_coeffs"],
]


@pytest.mark.parametrize("mode", list(GradeMode))
@pytest.mark.parametrize("optimized", _optimized)
def test_grader(optimized: list[str], mode: GradeMode, data_path: Path) -> None:
    """Test if `Grader` works."""
    path = data_path / "fitting/crystals/multi/10"
    images_training = read_cfg(path / "out.cfg", index=":")
    mtp_data = read_mtp(path / "pot.mtp")
    mtp_data.optimized = optimized

    grader = Grader(mtp_data, mode=mode)
    grader.update(images_training)
    images_out = grader.grade(images_training)

    assert all("MV_grade" in atoms.calc.results for atoms in images_out)


def test_comparision_with_mlip(data_path: Path) -> None:
    """Test if `Grader` gives the extrapolation grades consistent with MLIP."""
    path = data_path / "fitting/crystals/multi/10"
    images_training = read_cfg(path / "out.cfg", index=":")
    mtp_data = read_mtp(path / "pot.mtp")
    mtp_data.optimized = ["moment_coeffs", "species_coeffs", "radial_coeffs"]

    grades_ref = [atoms.calc.results["MV_grade"] for atoms in images_training]

    grader = Grader(mtp_data, mode="neighborhood", maxvol_setting={"algorithm": "mlip"})
    grader.update(images_training)
    images_out = grader.grade(images_training)

    grades = [atoms.calc.results["MV_grade"] for atoms in images_out]

    np.testing.assert_allclose(grades, grades_ref, rtol=1e-6, atol=1e-6)
