"""Tests for `motep grade`."""

import shutil
import subprocess
from contextlib import chdir
from pathlib import Path

from motep.grade.grader import load_setting_grade


def test_cli_grade(data_path: Path, tmp_path: Path) -> None:
    """Test `motep grade`."""
    with chdir(tmp_path):
        shutil.copy2(data_path / "original/molecules/291/training.cfg", ".")
        shutil.copy2(data_path / "fitting/molecules/291/02/pot.mtp", "final.mtp")
        shutil.copy2("training.cfg", "in.cfg")
        Path("motep.grade.toml").touch()  # empty
        args = ["motep", "grade", "motep.grade.toml"]
        result = subprocess.run(args, check=False)
        assert result.returncode == 0


def test_example_grade(doc_path: Path) -> None:
    """Test if the input file offered in the documentation is parsable."""
    path = doc_path / "cli/motep.grade.toml"
    setting = load_setting_grade(path)
    assert not isinstance(setting.potentials, dict)  # converted to a dataclass?
    assert isinstance(setting.potentials.final, str)
