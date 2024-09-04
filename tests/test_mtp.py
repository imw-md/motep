"""Tests for `mtp.py`."""

from motep.io.mlip.mtp import read_mtp, write_mtp


def test_mtp():
    """Test consistency between `read_mtp` and `write_mtp`."""
    d_ref = read_mtp("02.mtp")
    write_mtp("test.mtp", d_ref)
    d = read_mtp("test.mtp")
    assert d == d_ref
