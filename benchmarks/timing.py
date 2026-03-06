"""Contains functions (including a main function) for benchmarking."""

import argparse
import pathlib
import shutil
from time import perf_counter

import mpi4py
import numba as nb
import numpy as np
from ase import Atoms

from motep.calculator import MTP
from motep.io.mlip.cfg import read_cfg
from motep.io.mlip.mtp import read_mtp

comm = mpi4py.MPI.COMM_WORLD

fmt = "{:20s}"

setup_map = {
    "numpy": {"engine": "numpy"},
    "numba": {"engine": "numba"},
    "numba_train": {"engine": "numba", "mode": "train"},
    "jax": {"engine": "jax"},
    "cext": {"engine": "cext"},
    "cext_train": {"engine": "cext", "mode": "train"},
}

all_setups = ["numpy", "numba", "numba_train", "jax", "cext", "cext_train"]


def print_num_threads():
    if comm.rank == 0:
        print()
        print(f"Running benchmarks with {nb.get_num_threads()} threads.\n", flush=True)


class Timer:
    def __init__(self, name: str = "", *, verbose: bool = True) -> None:
        self.name: str = name
        self.start: float = float("nan")
        self.time: float = float("nan")
        self.print: bool = verbose

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start
        name = " " + self.name if self.name != "" else ""
        readout = f"Time{name}: {self.time * 1000:.3f} ms"
        if self.print and comm.rank == 0:
            print(readout, flush=True)


def _init_mlippy(pot_path: pathlib.Path, atom_number_list: list[int]):
    import mlippy  # noqa: PLC0415

    tmp_path = pathlib.Path("/tmp/motep_benchmarks/")
    tmp_path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pot_path, tmp_path / "pot.mtp")
    pot = mlippy.mtp(str(tmp_path / "pot.mtp"))
    for atomic_number in atom_number_list:
        pot.add_atomic_type(atomic_number)

    calc = mlippy.MLIP_Calculator(pot, {})
    calc.use_cache = False
    return calc


def _time_mlippy(pot_path: pathlib.Path, images: list[Atoms]) -> np.ndarray:
    atom_number_list = []
    for n in images[0].get_atomic_numbers():
        if n not in atom_number_list:
            atom_number_list.append(n)
    calc = _init_mlippy(pot_path, atom_number_list)
    # Make initial calc to not time things like compile time and things that are cachable
    calc.get_potential_energy(images[-1])
    with Timer(fmt.format("mlippy (run)")):
        energies = [calc.get_potential_energy(_) for _ in images]
    return np.array(energies)


def _time_mtp(
    pot_path: pathlib.Path,
    images: list[Atoms],
    *,
    engine: str,
    mode: str = "run",
) -> np.ndarray:
    mtp_data = read_mtp(pot_path)
    mtp_data.species = []
    for atomic_number in images[0].numbers:
        if atomic_number not in mtp_data.species:
            mtp_data.species.append(atomic_number)
    calc = MTP(mtp_data, engine=engine, mode=mode)
    calc.use_cache = False

    suffix = f" ({mode})"

    # Make initial calc to not time things like compile time and things that are cachable
    with Timer(fmt.format(engine + suffix + " (0th)")):
        calc.get_potential_energy(images[-1])

    comm.barrier()
    with Timer(fmt.format(engine + suffix)):
        energies = [calc.get_potential_energy(_) for _ in images]
        comm.barrier()
    return np.array(energies)


def main(
    setup_names: list[str],
    levels: list[int] | None = None,
    sizes: list[int] | None = None,
    nimages: int | None = None,
) -> None:
    """Run benchmarks."""
    print_num_threads()
    setups = [setup_map[_] for _ in setup_names or all_setups]
    data_path = pathlib.Path(__file__).parent / "../tests/data_path"
    crystal = "cubic"
    cfg_path = data_path / f"original/crystals/{crystal}/training.cfg"
    index = slice(0, nimages or 10)
    orig_images = read_cfg(cfg_path, index=index)
    for level in levels or [12, 18, 24]:
        for size_reps in sizes or [1, 3]:
            images = [_.repeat(size_reps) for _ in orig_images]
            number_of_atoms = len(images[0])
            if comm.rank == 0:
                print(
                    f"\nTiming for {len(images)} images"
                    f" of {number_of_atoms} atoms"
                    f" with level {level}"
                    f" on {comm.size} ranks:",
                    flush=True,
                )

            path = data_path / f"fitting/crystals/{crystal}/{level:02d}"
            pot_path = path / "pot.mtp"

            try:
                _time_mlippy(pot_path, images) if comm.rank == 0 else None
            except ImportError:
                if comm.rank == 0:
                    print("mlippy could not be imported, skipping mlippy ref.")

            for setup in setups:
                _time_mtp(pot_path, images, **setup)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("setups", nargs="*", choices=all_setups)
    parser.add_argument("--levels", nargs="+", choices=list(range(2, 27, 2)), type=int)
    parser.add_argument("--sizes", nargs="+", type=int)
    parser.add_argument("--nimages", type=int)
    args = parser.parse_args()
    main(args.setups, args.levels, args.sizes, args.nimages)
