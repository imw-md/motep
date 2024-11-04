"""Tests for `mtp.py`."""

import pathlib

import numpy as np
import pytest

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


@pytest.mark.parametrize("level", [2, 4])
def test_alpha_index_times(level: int, data_path: pathlib.Path) -> None:
    """Test shape of `alpha_index_times`."""
    path = data_path / f"fitting/molecules/291/{level:02d}"
    mtp_data = read_mtp(path / "initial.mtp")
    assert mtp_data["alpha_index_times"].ndim == 2
    assert mtp_data["alpha_index_times"].shape[-1] == 4
