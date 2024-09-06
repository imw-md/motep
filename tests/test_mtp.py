"""Tests for `mtp.py`."""

from motep.io.mlip.mtp import read_mtp, write_mtp


def test_mtp():
    """Test consistency between `read_mtp` and `write_mtp`."""
    d_ref = read_mtp("initial.mtp")
    write_mtp("final.mtp", d_ref)
    d = read_mtp("final.mtp")
    assert d == d_ref
