"""Tests for `mtp.py`."""

import pathlib
from dataclasses import asdict

import numpy as np
import pytest

from motep.io.mlip.mtp import read_mtp, write_mtp
from motep.potentials.mtp.data import BasisData, MTPData


def test_mtp(data_path: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """Test consistency between `read_mtp` and `write_mtp`."""
    path = data_path / "fitting/molecules/291/02"
    d_ref = read_mtp(path / "initial.mtp")
    write_mtp(tmp_path / "final.mtp", d_ref)
    d = read_mtp(tmp_path / "final.mtp")

    d_ref = asdict(d_ref)
    d = asdict(d)

    for key, value in d_ref.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(d[key], value)
        else:
            assert d[key] == value


@pytest.mark.parametrize("legacy", [False, True])
def test_mtp_roundtrip_initialized(legacy: bool, tmp_path: pathlib.Path) -> None:
    """Roundtrip write/read test with initialized (non-None) radial_coeffs."""
    data = MTPData(
        version="1.1.0",
        species_count=2,
        radial_funcs_count=4,
        radial_basis=BasisData(type="RBChebyshev", min=2.0, max=5.0, size=8),
        alpha_scalar_moments=3,
    )
    data.initialize(np.random.default_rng(0))

    write_mtp(tmp_path / "test.mtp", data, legacy=legacy)
    d = read_mtp(tmp_path / "test.mtp")

    d_ref = asdict(data)
    d = asdict(d)

    for key, value in d_ref.items():
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating):
            np.testing.assert_allclose(d[key], value, rtol=1e-14)
        elif isinstance(value, np.ndarray):
            np.testing.assert_array_equal(d[key], value)
        else:
            assert d[key] == value


@pytest.mark.parametrize("level", [2, 4])
def test_alpha_index_times(level: int, data_path: pathlib.Path) -> None:
    """Test shape of `alpha_index_times`."""
    path = data_path / f"fitting/molecules/291/{level:02d}"
    mtp_data = read_mtp(path / "initial.mtp")
    assert mtp_data.alpha_index_times.ndim == 2
    assert mtp_data.alpha_index_times.shape[-1] == 4
