"""Tests for `motep train`."""

import shutil
import subprocess
from contextlib import chdir
from pathlib import Path

import pytest

from motep.train.setting import _Setting
from motep.train.trainer import load_setting_train


def test_cli_train(data_path: Path, tmp_path: Path) -> None:
    """Test `motep train`."""
    with chdir(tmp_path):
        shutil.copy2(data_path / "original/molecules/291/training.cfg", ".")
        shutil.copy2(data_path / "fitting/molecules/291/02/initial.mtp", ".")
        Path("motep.train.toml").touch()  # empty
        args = ["motep", "train", "motep.train.toml"]
        result = subprocess.run(args, check=False)
        assert result.returncode == 0


def test_example_train(doc_path: Path) -> None:
    """Test if the input file offered in the documentation is parsable."""
    path = doc_path / "cli/motep.train.toml"
    setting = load_setting_train(path)
    assert not isinstance(setting.configurations, dict)  # converted to a dataclass?
    assert isinstance(setting.configurations.training, list)
    assert not isinstance(setting.potentials, dict)  # converted to a dataclass?
    assert isinstance(setting.potentials.final, str)


@pytest.mark.parametrize(
    "steps",
    [
        [{"method": "minimize"}],
        [{"method": "l-bfgs-b"}],
        [{"method": "minimize", "kwargs": {"method": "l-bfgs-b"}}],
    ],
)
def test_train_steps_setting(steps: list) -> None:
    """Test if `steps` in `TrainSetting` is correctly parsed."""
    setting = _Setting(steps=steps)
    assert setting.steps[0]["method"] == "minimize"
