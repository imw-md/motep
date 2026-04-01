"""Tests for `motep evaluate`."""

import shutil
import subprocess
from contextlib import chdir
from pathlib import Path

from motep.evaluate.evaluator import load_setting_evaluate


def test_cli_evaluate(data_path: Path, tmp_path: Path) -> None:
    """Test `motep evaluate`."""
    with chdir(tmp_path):
        shutil.copy2(data_path / "original/molecules/291/training.cfg", "initial.cfg")
        shutil.copy2(data_path / "fitting/molecules/291/02/pot.mtp", "final.mtp")
        Path("motep.evaluate.toml").touch()  # empty
        args = ["motep", "evaluate", "motep.evaluate.toml"]
        result = subprocess.run(args, check=False)
        assert result.returncode == 0


def test_example_evaluate(doc_path: Path) -> None:
    """Test if the input file offered in the documentation is parsable."""
    path = doc_path / "cli/motep.evaluate.toml"
    setting = load_setting_evaluate(path)
    assert not isinstance(setting.common, dict)  # converted to a dataclass?
    assert not isinstance(setting.configurations, dict)  # converted to a dataclass?
    assert isinstance(setting.configurations.initial, list)
    assert not isinstance(setting.potentials, dict)  # converted to a dataclass?
    assert isinstance(setting.potentials.final, str)
