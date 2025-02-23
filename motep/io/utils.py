"""IO unils."""

from ase import Atoms
from mpi4py import MPI

import motep.io


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
    rank = comm.Get_rank()
    if rank == 0:
        print(f"{'':=^72s}\n")
        print(f"[{title}]")
        images = []
        for filename in filenames:
            images_local = motep.io.read(filename, species)
            images.extend(images_local)
            print(f'"{filename}" = {len(images_local)}')
        print()
    return comm.bcast(images, root=0)
