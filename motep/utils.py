"""Utilities."""

import contextlib
import os
import pathlib
import typing


@contextlib.contextmanager
def cd(path: str | pathlib.Path) -> typing.Generator:
    """Change directory temporalily.

    Parameters
    ----------
    path: Path
        Path to directory.

    """
    cwd = pathlib.Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(cwd)
