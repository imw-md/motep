"""Tests for `mtp.py`."""

import pathlib

from motep.io.mlip.mtp import read_mtp, write_mtp


def test_mtp(data_path: pathlib.Path, tmp_path: pathlib.Path):
    """Test consistency between `read_mtp` and `write_mtp`."""
    path = data_path / f"fitting/molecules/291/02"
    d_ref = read_mtp(path / "initial.mtp")
    write_mtp(tmp_path / "final.mtp", d_ref)
    d = read_mtp(tmp_path / "final.mtp")
    assert d == d_ref
