"""Tests for CLI."""

import shutil
import subprocess
import sys

import pytest


def test_help_main() -> None:
    """Test `motep -h`."""
    motep = shutil.which("motep")
    assert motep is not None

    args = [motep, "-h"]
    result = subprocess.run(args, check=False)
    assert result.returncode == 0

    args = [sys.executable, "-m", "motep", "-h"]
    result = subprocess.run(args, check=False)
    assert result.returncode == 0


@pytest.mark.parametrize("command", ["train", "grade"])
def test_help_sub(command: str) -> None:
    """Test `motep command -h`."""
    motep = shutil.which("motep")
    assert motep is not None

    args = [motep, command, "-h"]
    result = subprocess.run(args, check=False)
    assert result.returncode == 0

    args = [sys.executable, "-m", f"motep.{command}", "-h"]
    result = subprocess.run(args, check=False)
    assert result.returncode == 0
