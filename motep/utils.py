"""Utilities."""

import contextlib
import os
import pathlib
import time
import typing

from motep.parallel import DummyMPIComm, world


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


@contextlib.contextmanager
def measure_time(name: str, *, comm: DummyMPIComm = world) -> typing.Generator:
    """Measure time.

    Parameters
    ----------
    name : str
        Name of the block.
    comm : MPI.Comm
        MPI communicator.

    """
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        if comm.rank == 0:
            print(f"Time ({name}): {end_time - start_time} (s)\n", flush=True)
