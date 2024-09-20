"""Tests for trainer.py."""

import pathlib

from mpi4py import MPI

from motep.io.mlip.cfg import read_cfg
from motep.mlippy_loss_function import calc_properties


def test_current_value() -> None:
    """Test `current_value`."""
    fn = pathlib.Path(__file__).parent / "test.cfg"
    images = read_cfg(fn, index=":")
    calc_properties(images, MPI.COMM_WORLD)
