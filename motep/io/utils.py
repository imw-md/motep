"""IO unils."""

import logging

from ase import Atoms

import motep.io
from motep.parallel import DummyMPIComm, world

logger = logging.getLogger(__name__)


def get_dummy_species(images: list[Atoms]) -> list[int]:
    """Get dummy species particularly for images read from `.cfg` files.

    Returns
    -------
    list[int]

    """
    m = 0
    for atoms in images:
        m = max(m, atoms.numbers.max())
    return list(range(m + 1))


def read_images(
    filenames: list[str],
    species: list[int] | None = None,
    comm: DummyMPIComm = world,
    title: str = "configurations",
) -> list[Atoms]:
    """Read images.

    Returns
    -------
    list[Atoms]

    """
    images = []
    if comm.rank == 0:
        logger.info("%s\n", "=" * 72)
        logger.info("[%s]", title)
        for filename in filenames:
            images_local = motep.io.read(filename, species)
            images.extend(images_local)
            logger.info('"%s" = %s', filename, len(images_local))
        logger.info("")
    return comm.bcast(images, root=0)
