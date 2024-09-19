import sys

sys.path.append("./")

import os
import pathlib
import shutil
from time import perf_counter

import ase.atoms
import ase.io
import mlippy
import numpy as np

from motep.calculator import MTP
from motep.io.mlip.cfg import read_cfg
from motep.io.mlip.mtp import read_mtp


class Timer:
    def __init__(self, name="", print=True):
        self.name = name
        self.print = print

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start
        name = " " + self.name if self.name != "" else ""
        self.readout = f"Time{name}: {self.time * 1000:.3f} ms"
        if self.print:
            print(self.readout)


def init_mlippy(pot_path: pathlib.Path):
    tmp_path = pathlib.Path("/tmp/motep_benchmarks/")
    os.makedirs(tmp_path, exist_ok=True)
    shutil.copy2(pot_path, tmp_path / "pot.mtp")
    pot = mlippy.mtp(str(tmp_path / "pot.mtp"))
    for _ in range(1):
        pot.add_atomic_type(_)

    calc = mlippy.MLIP_Calculator(pot, {})
    return calc


def time_mlippy(
    pot_path: pathlib.Path,
    images: list[ase.atoms.Atoms],
) -> None:
    calc = init_mlippy(pot_path)
    # Make initial calc to not time things like compile time and things that are cachable
    calc.get_potential_energy(images[-1])
    with Timer("mlippy"):
        np.array([calc.get_potential_energy(_) for _ in images])


def time_mtp(
    pot_path: pathlib.Path,
    images: list[ase.atoms.Atoms],
    engine: str,
) -> None:
    parameters = read_mtp(pot_path)
    calc = MTP(parameters, engine=engine)
    # Make initial calc to not time things like compile time and things that are cachable
    calc.get_potential_energy(images[-1])
    with Timer(engine):
        np.array([calc.get_potential_energy(_) for _ in images])


def time_numpy(
    pot_path: pathlib.Path,
    images: list[ase.atoms.Atoms],
) -> None:
    time_mtp(pot_path, images, "numpy")


def time_numba(
    pot_path: pathlib.Path,
    images: list[ase.atoms.Atoms],
) -> None:
    time_mtp(pot_path, images, "numba")


if __name__ == "__main__":
    data_path = pathlib.Path("tests/data_path")
    crystal = "cubic"
    for level in [2, 4, 6, 8, 10]:
        path = data_path / f"fitting/crystals/{crystal}/{level:02d}"
        cfg_path = path / "out.cfg"
        if not cfg_path.is_file():
            continue
        index = slice(0, 10)  # ":"
        images = read_cfg(cfg_path, index=index)
        print(f"Timing for {len(images)} images with level {level}:")
        time_mlippy(path / "pot.mtp", images)
        time_numpy(path / "pot.mtp", images)
        time_numba(path / "pot.mtp", images)
        print()
