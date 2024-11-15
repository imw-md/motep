"""IO."""

import ase.io
from ase import Atoms

from motep.io.mlip.cfg import read_cfg


def read(filenames: list[str], species: list[int] | None = None) -> list[Atoms]:
    """Read images."""
    images = []
    for filename in filenames:
        if isinstance(filename, str) and filename.endswith(".cfg"):
            images.extend(read_cfg(filename, index=":", species=species))
        else:
            images.extend(ase.io.read(filename, index=":"))
    return images
