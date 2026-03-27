"""Common setting for pytest."""

from pathlib import Path

import pytest


@pytest.fixture
def data_path() -> Path:
    """Get path to the MD-trajectory data.

    Returns
    -------
    Path

    """
    return Path(__file__).parent / "data_path"


@pytest.fixture
def doc_path() -> Path:
    """Get path to the documentation.

    Returns
    -------
    Path

    """
    return Path(__file__).parents[1] / "docs"
