"""Tests for IO."""

import pathlib

import numpy as np

import motep.io
from motep.io.mlip.cfg import read_cfg, write_cfg
from motep.trainer import read_images


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
    images = motep.io.read(str(path) + "@0")
    assert len(images) == 1
    images = motep.io.read(str(path) + "@0:2")
    assert len(images) == 2


def test_read_multiple_files(data_path: pathlib.Path) -> None:
    """Test if multiple files can be read."""
    configurations = []
    for molecule in [762, 291]:
        path = data_path / f"original/molecules/{molecule}/training.cfg"
        configurations.append(str(path))
    n_ref = sum(len(read_cfg(_, index=":")) for _ in configurations)
    assert len(read_images(configurations)) == n_ref


def test_read_ase_file(data_path: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """Test if the ASE-recognized file can be read."""
    molecule = 762
    path = data_path / f"original/molecules/{molecule}/training.cfg"
    atoms = read_cfg(path)
    fd = tmp_path / "test.xyz"
    atoms.write(fd)
    assert motep.io.read(fd)


def test_roundtrip(data_path: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """Test if `write_cfg` works as expected."""
    molecule = 762
    path = data_path / f"original/molecules/{molecule}/training.cfg"
    atoms_ref = read_cfg(path)
    atoms_ref.calc.results.pop("free_energy")
    fd = tmp_path / "test.cfg"
    write_cfg(fd, atoms_ref)
    atoms = read_cfg(fd)
    assert atoms == atoms_ref
    for k, v in atoms_ref.calc.results.items():
        assert np.all(atoms.calc.results[k] == v)
