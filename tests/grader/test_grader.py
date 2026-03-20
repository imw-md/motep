"""Tests for active-learning algorithms."""

from pathlib import Path

import numpy as np
import pytest

from motep.grader.grader import GradeMode, Grader
from motep.io.mlip.cfg import read_cfg
from motep.io.mlip.mtp import read_mtp


@pytest.mark.parametrize("mode", list(GradeMode))
def test_grader(mode: GradeMode, data_path: Path) -> None:
    """Test if `Grader` works."""
    path = data_path / "fitting/crystals/multi/10"
    images_training = read_cfg(path / "out.cfg", index=":")
    mtp_data = read_mtp(path / "pot.mtp")

    grades_ref = [atoms.calc.results["MV_grade"] for atoms in images_training]

    grader = Grader(mtp_data, mode=mode)
    grader.update(images_training)
    images_out = grader.grade(images_training)

    grades = [atoms.calc.results["MV_grade"] for atoms in images_out]

    # np.testing.assert_allclose(grades, grades_ref)  # TODO
