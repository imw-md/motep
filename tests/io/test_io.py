"""Tests for IO."""

import pathlib

from ase import Atoms

import motep.io
from motep.io.mlip.cfg import read_cfg


def test_read_path(data_path: pathlib.Path) -> None:
    """Test if the `pathlib.Path` object can be read directly."""
    molecule = 762
    path = data_path / f"original/molecules/{molecule}/training.cfg"
    read_cfg(path)


def test_index(data_path: pathlib.Path) -> None:
    """Test if `read_cfg` can accept a flexible `index`."""
    molecule = 762
    path = data_path / f"original/molecules/{molecule}/training.cfg"
    images = read_cfg(path, index="0:2")
    assert len(images) == 2


def test_parse_filename(data_path: pathlib.Path) -> None:
    """Test if the ASE at-mark syntax works."""
    molecule = 762
    path = data_path / f"original/molecules/{molecule}/training.cfg"
    images = motep.io.read([str(path) + "@0"])
    assert len(images) == 1
    images = motep.io.read([str(path) + "@0:2"])
    assert len(images) == 2


def test_read_multiple_files(data_path: pathlib.Path) -> None:
    """Test if multiple files can be read."""
    configurations = []
    for molecule in [762, 291]:
        path = data_path / f"original/molecules/{molecule}/training.cfg"
        configurations.append(str(path))
    n_ref = sum(len(read_cfg(_, index=":")) for _ in configurations)
    assert len(motep.io.read(configurations)) == n_ref


def test_read_ase_file(data_path: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """Test if the ASE-recognized file can be read."""
    molecule = 762
    path = data_path / f"original/molecules/{molecule}/training.cfg"
    atoms = read_cfg(path)
    fd = tmp_path / "test.xyz"
    atoms.write(fd)
    assert motep.io.read([fd])
