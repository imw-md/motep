"""IO."""

import ase.io
from ase import Atoms
from ase.io.formats import parse_filename

from motep.io.mlip.cfg import read_cfg


def read(filenames: list[str], species: list[int] | None = None) -> list[Atoms]:
    """Read images."""
    images = []
    for filename in filenames:
        filename_parsed, index = parse_filename(filename)
        index = ":" if index is None else index
        if isinstance(filename_parsed, str) and filename_parsed.endswith(".cfg"):
            atoms = read_cfg(filename_parsed, index=index, species=species)
        else:
            atoms = ase.io.read(filename_parsed, index=index)
        if isinstance(atoms, Atoms):
            images.append(atoms)
        else:
            images.extend(atoms)
    return images
