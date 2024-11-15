"""IO."""

from ase import Atoms

from motep.io.mlip.cfg import read_cfg


def read(filenames: list[str], species: list[int] | None = None) -> list[Atoms]:
    """Read images."""
    images = []
    for filename in filenames:
        images.extend(read_cfg(filename, index=":", species=species))
    return images
