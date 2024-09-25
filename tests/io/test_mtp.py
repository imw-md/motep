"""Tests for `mtp.py`."""

import pathlib

import numpy as np

from motep.io.mlip.mtp import read_mtp, write_mtp


def test_mtp(data_path: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """Test consistency between `read_mtp` and `write_mtp`."""
    path = data_path / "fitting/molecules/291/02"
    d_ref = read_mtp(path / "initial.mtp")
    write_mtp(tmp_path / "final.mtp", d_ref)
    d = read_mtp(tmp_path / "final.mtp")
    for key, value in d_ref.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(d[key], value)
        else:
            assert d[key] == value
