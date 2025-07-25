"""Module for MTP formats."""

import pathlib
from typing import TextIO

import numpy as np
from ase import Atoms
from ase.calculators.lammps import convert
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import atomic_masses, chemical_symbols
from ase.utils import reader, string2index, writer


@reader
def read_cfg(
    fd: pathlib.Path,
    index: int = -1,
    species: list[int] | list[str] | None = None,
) -> Atoms | list[Atoms]:
    """Read images from a MTP .cfg file.

    Parameters
    ----------
    species : list[int] | list[str], optional
        List defining types of chemical symbols. For example,
        [46, 1] and ['Pd', 'H'] assign Pd for type 0 and H for type 1. If None,
        dummy symbols 'X', 'H', 'He', etc. are assigned for types 0, 1, 2, etc.

    Returns
    -------
    Atoms | list[Atoms]

    """
    species = _convert_species(species)
    atoms_list = []
    for line in fd:
        if line.startswith("BEGIN_CFG"):
            atoms = _read_image(fd, species)
            atoms_list.append(atoms)

    if isinstance(index, str):
        index = string2index(index)

    return atoms_list[index]


def _convert_species(species: list | None) -> list[int] | None:
    if isinstance(species, list) and isinstance(species[0], str):
        return [chemical_symbols.index(_) for _ in species]
    return species  # list[int] | None


def _read_image(file: TextIO, species: list[str] | None) -> Atoms:
    keys_c = ["cartes_x", "cartes_y", "cartes_z"]
    keys_d = ["direct_x", "direct_y", "direct_z"]
    cell = None
    stress = None
    results = {}
    info = {}  # added PK
    for line in file:
        if line.startswith("END_CFG"):
            break
        if line.split()[0] == "Size":
            size = int(next(file).split()[0])
        elif line.split()[0] in {"Supercell", "SuperCell"}:
            cell = [[float(_) for _ in next(file).split()] for _ in range(3)]
        elif line.split()[0] in {"AtomData:", "Atomic_data:"}:
            atomdata = {_: [] for _ in line.split()[1:]}
            for _ in range(size):
                for key, value in zip(atomdata, next(file).split(), strict=True):
                    atomdata[key].append(_parse_value(value))
        elif line.split()[0] == "Energy":
            energy = float(next(file).split()[0])
            for k in ["energy", "free_energy"]:
                results[k] = energy
        elif line.split()[0] == "PlusStress:":
            keys = line.split()[1:]
            stress = [float(value) for value in next(file).split()]
            stress = dict(zip(keys, stress, strict=True))
        elif line.split()[0] == "Feature":
            try:
                info[str(line.split()[1])] = float(line.split()[2])
            except ValueError:
                info[str(line.split()[1])] = line.split()[2]

    if species is None:
        numbers = atomdata["type"]
    else:
        numbers = [species[_] for _ in atomdata["type"]]

    pbc = cell is not None

    if all((_ in atomdata) for _ in keys_c):
        positions = list(zip(*[atomdata[_] for _ in keys_c], strict=True))
        atoms = Atoms(
            numbers=numbers,
            positions=positions,
            cell=cell,
            pbc=pbc,
        )
    elif all((_ in atomdata) for _ in keys_d):
        positions = list(zip(*[atomdata[_] for _ in keys_d], strict=True))
        atoms = Atoms(
            numbers=numbers,
            scaled_positions=positions,
            cell=cell,
            pbc=pbc,
        )
    else:
        raise ValueError

    for key, value in atomdata.items():
        if key in {"id", "type"} or key in keys_c or key in keys_d:
            continue
        results[key] = np.array(value)

    atoms.calc = SinglePointCalculator(atoms)
    atoms.calc.results.update(results)
    if "fx" in atomdata:
        _set_forces(atoms, atomdata)
    if cell is not None and stress is not None:
        _set_stress(atoms, stress)
    atoms.info = info  # added PK
    return atoms


def _set_forces(atoms: Atoms, atomdata: dict) -> None:
    keys_forces = ["fx", "fy", "fz"]
    forces = list(zip(*[atomdata[_] for _ in keys_forces], strict=True))
    atoms.calc.results["forces"] = np.array(forces)


def _set_stress(atoms: Atoms, stress: dict[float]) -> None:
    voigt_order = ["xx", "yy", "zz", "yz", "xz", "xy"]
    stress = np.array([stress[_] for _ in voigt_order])
    atoms.calc.results["stress"] = -stress / atoms.get_volume()


def _parse_value(value: str) -> int | float | bool:
    if value.isdigit():
        return int(value)
    if value.replace(".", "").replace("-", "").isdigit():
        return float(value)
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    raise RuntimeError(value)


@writer
def write_cfg(
    fd: pathlib.Path,
    images: Atoms | list[Atoms],
    species: list[int] | list[str] | None = None,
    key_energy: str | None = None,
) -> None:
    """Write images into the MTP .cfg format.

    Parameters
    ----------
    filename : `pathlib.Path`
        _description_
    images : Atoms | list[Atoms]
        _description_
    species : list[int] | list[str], optional
        List that defines types of chemical symbols (e.g, ['Pd', 'H'] means Pd
        is type 0 and H type 1), by default None. If None, this list is built
        by assigning each distinct species to an integer in the order of
        appearance in `images`.
    key_energy : str | None, default: None
        Key for the energy (either "free_energy" or "energy") to be printed.
        If :py:obj:`None`, "free_energy" is first considered, followed by "energy".

    """
    if isinstance(images, Atoms):
        images = [images]

    species = _convert_species(species)
    if species is None:
        species = _get_species(images)

    keys_energy = ["free_energy", "energy"] if key_energy is None else [key_energy]

    for atoms in images:
        _write_image(fd, atoms, species, keys_energy)


def _get_species(images: list[Atoms]) -> list[int]:
    numbers = []
    for atoms in images:
        numbers.extend(atoms.get_atomic_numbers())
    return sorted(set(numbers), key=numbers.index)


def _write_image(
    file: TextIO,
    atoms: Atoms,
    species: list[int],
    keys_energy: list[str],
) -> None:
    if not hasattr(atoms, "calc") or atoms.calc is None:
        atoms.calc = SinglePointCalculator(atoms)  # dummy calculator

    file.write("BEGIN_CFG\n")
    file.write(" Size\n")
    file.write(f"{len(atoms):6d}\n")

    if all(atoms.pbc):
        _write_supercell(file, atoms)

    _write_atom_data(file, atoms, species)

    for key_energy in keys_energy:
        if key_energy in atoms.calc.results:
            energy = atoms.calc.get_property(key_energy)
            file.write(" Energy\n")
            file.write(f"{energy:24.12f}\n")
            break

    if "stress" in atoms.calc.results:
        _write_stress(file, atoms)
    for key in atoms.info:
        file.write(f" Feature   {key}\t{atoms.info[key]}\n")
    file.write("END_CFG\n")
    file.write("\n")


def _write_supercell(file: TextIO, atoms: Atoms) -> None:
    file.write(" Supercell\n")
    for vector in atoms.cell:
        file.write("   ")
        for _ in vector:
            file.write(f"{_:14.6f}")
        file.write("\n")


def _write_atom_data(file: TextIO, atoms: Atoms, species: list[int]) -> None:
    line = " AtomData:  id type "
    file.write(line)
    for _ in ["cartes_x", "cartes_y", "cartes_z"]:
        file.write(f"{_:>14s}")
    if "forces" in atoms.calc.results:
        file.write(" ")
        for _ in ["fx", "fy", "fz"]:
            file.write(f"{_:>12s}")
    file.write("\n")
    numbers = atoms.get_atomic_numbers()
    positions = atoms.get_positions()
    if "forces" in atoms.calc.results:
        forces = atoms.calc.results["forces"]
    for i, number in enumerate(numbers):
        file.write(f"{i + 1:14d}")
        file.write(f"{species.index(number):5d}")
        file.write(" ")
        for j in range(3):
            file.write(f"{positions[i, j]:14.6f}")
        if "forces" in atoms.calc.results:
            file.write(" ")
            for j in range(3):
                file.write(f"{forces[i, j]:12.6f}")
        file.write("\n")


def _write_stress(file: TextIO, atoms: Atoms) -> None:
    line = "".join([f"{_:>12s}" for _ in ["xx", "yy", "zz", "yz", "xz", "xy"]])
    line = f" PlusStress:{line[8:]}\n"
    file.write(line)
    file.write("    ")
    for _ in atoms.get_stress():
        _ *= -1.0 * atoms.get_volume()
        file.write(f"{_:12.5f}")
    file.write("\n")


def _write_parameters(file: TextIO, species: dict[str, int]) -> None:
    file.write("# Masses\n\n")
    units = "metal"  # g/mol
    for s, i in species.items():
        atomic_number = chemical_symbols.index(s)
        mass = atomic_masses[atomic_number]
        mass = convert(mass, "mass", "ASE", units)
        atom_type = i + 1
        file.write(f"mass {atom_type:>6} {mass:23.17g} # {s}\n")
