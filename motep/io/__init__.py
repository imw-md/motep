"""IO."""

import ase.io
from ase import Atoms
from ase.io.formats import parse_filename

from motep.io.mlip.cfg import read_cfg, write_cfg


def read(filename: str, species: list[int] | None = None) -> list[Atoms]:
    """Read images.

    Parameters
    ----------
    filename : str
        File name to be read.
        Both the MLIP `.cfg` format and the ASE-recognized formats can be parsed.

        To select a part of images, the ASE `@` syntax can be used as follows.

        https://wiki.fysik.dtu.dk/ase/ase/gui/basics.html#selecting-part-of-a-trajectory

        - `x.traj@0:10:1`: first 10 images
        - `x.traj@0:10`: first 10 images
        - `x.traj@:10`: first 10 images
        - `x.traj@-10:`: last 10 images
        - `x.traj@0`: first image
        - `x.traj@-1`: last image
        - `x.traj@::2`: every second image

        Further, for the ASE database format, i.e., `.json` and `.db`,
        the extended ASE syntax can also be used as follows.

        https://wiki.fysik.dtu.dk/ase/ase/db/db.html#integration-with-other-parts-of-ase

        https://wiki.fysik.dtu.dk/ase/ase/db/db.html#querying

        - `x.db@H>0`: images with hydrogen atoms

    species : list[int]
        List of atomic numbers for the atomic types in the MLIP `.cfg` format.

    Returns
    -------
    list[Atoms]
        List of ASE `Atoms` objects.

    """
    filename_parsed, index = parse_filename(filename)
    index = ":" if index is None else index
    if isinstance(filename_parsed, str) and filename_parsed.endswith(".cfg"):
        images = read_cfg(filename_parsed, index=index, species=species)
    else:
        images = ase.io.read(filename_parsed, index=index, parallel=False)
    return [images] if isinstance(images, Atoms) else images


def write(filename: str, images: list[Atoms], species: list[int] | None = None) -> None:
    """Write images.

    Parameters
    ----------
    filename : str
        File name to be written.
        Both the MLIP `.cfg` format and the ASE-recognized formats can be parsed.
    images : list[Atoms]
        List of ASE `Atoms` objects.
    species : list[int]
        List of atomic numbers for the atomic types in the MLIP `.cfg` format.

    """
    if filename.endswith(".cfg"):
        return write_cfg(filename, images, species=species)
    return ase.io.write(filename, images)
