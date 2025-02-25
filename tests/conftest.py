"""Common setting for pytest."""

from pathlib import Path

import pytest


@pytest.fixture
def data_path() -> Path:
    """Get path to the MD-trajectory data of molecules."""
    return Path(__file__).parent / "data_path"
