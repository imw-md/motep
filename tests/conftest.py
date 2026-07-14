"""Common setting for pytest."""

import logging
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


def pytest_configure(config) -> None:
    """Configure pytest."""
    logging.getLogger("motep").setLevel(logging.DEBUG)


def pytest_collection_modifyitems(items) -> None:
    """Set DEBUG level on all test_* loggers after collection."""
    for name in logging.root.manager.loggerDict:
        if name.startswith("test_"):
            logging.getLogger(name).setLevel(logging.DEBUG)
