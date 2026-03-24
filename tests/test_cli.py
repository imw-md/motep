"""Tests for CLI."""

import shutil
import subprocess
import sys


def test_help() -> None:
    """Test `motep -h`."""
    motep = shutil.which("motep")
    assert motep is not None

    result = subprocess.run([motep, "-h"], check=False)
    assert result.returncode == 0

    result = subprocess.run([sys.executable, "-m", "motep", "-h"], check=False)
    assert result.returncode == 0
