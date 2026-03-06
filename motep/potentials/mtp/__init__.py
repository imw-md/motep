import numpy as np
import numpy.typing as npt
from ase import Atoms


def get_types(atoms: Atoms, species: list[int]) -> npt.NDArray[np.int32]:
    return np.fromiter((species.index(_) for _ in atoms.numbers), dtype=np.int32)
