"""Common setting for pytest."""
import pathlib

import pytest


@pytest.fixture()
def data_path():
    """Data path to the MD trajectories of molecules."""
    return pathlib.Path(__file__).parent / "data_path"
