"""IO unils."""

import logging

from ase import Atoms
from mpi4py import MPI

import motep.io

logger = logging.getLogger(__name__)


def get_dummy_species(images: list[Atoms]) -> list[int]:
    """Get dummy species particularly for images read from `.cfg` files."""
    m = 0
    for atoms in images:
        m = max(m, atoms.numbers.max())
    return list(range(m + 1))


def read_images(
    filenames: list[str],
    species: list[int] | None = None,
    comm: MPI.Comm = MPI.COMM_WORLD,
    title: str = "data",
) -> list[Atoms]:
    """Read images."""
    images = []
    if comm.rank == 0:
        logger.info(f"{'':=^72s}\n")
        logger.info(f"[{title}]")
        for filename in filenames:
            images_local = motep.io.read(filename, species)
            images.extend(images_local)
            logger.info(f'"{filename}" = {len(images_local)}')
        logger.info("")
    return comm.bcast(images, root=0)
