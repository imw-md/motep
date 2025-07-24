"""Tests for upconverter.py."""

from pathlib import Path

import pytest

from motep.io.mlip.cfg import read_cfg
from motep.io.mlip.mtp import read_mtp
from motep.potentials.mtp.numba.engine import NumbaMTPEngine
from motep.upconverter import upconvert

_levels = [[i, j] for i in range(2, 22, 2) for j in range(i, 22, 2)]


@pytest.mark.parametrize(("src_level", "dst_level"), _levels)
def test_upconvert(src_level: int, dst_level: int, data_path: Path) -> None:
    """Test `upconvert`."""
    src_path = data_path / f"fitting/crystals/cubic/{src_level:02d}"
    dst_path = data_path / f"fitting/crystals/cubic/{dst_level:02d}"
    src_dat = read_mtp(src_path / "pot.mtp")
    dst_dat = read_mtp(dst_path / "pot.mtp")
    upconvert(src_dat, dst_dat)
    src_mtp = NumbaMTPEngine(src_dat, is_trained=False)
    dst_mtp = NumbaMTPEngine(dst_dat, is_trained=False)
    atoms = read_cfg(src_path / "out.cfg")
    src_energy = src_mtp.calculate(atoms)["energy"]
    dst_energy = dst_mtp.calculate(atoms)["energy"]
    assert dst_energy == pytest.approx(src_energy)
