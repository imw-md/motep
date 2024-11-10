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

fmt = "{:12s}"


class Timer:
    def __init__(self, name: str = "", print: bool = True) -> None:
        self.name: str = name
        self.start: float = float("nan")
        self.time: float = float("nan")
        self.print: bool = print

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start
        name = " " + self.name if self.name != "" else ""
        readout = f"Time{name}: {self.time * 1000:.3f} ms"
        if self.print:
            print(readout)


def init_mlippy(pot_path: pathlib.Path, atom_number_list: list[int]):
    tmp_path = pathlib.Path("/tmp/motep_benchmarks/")
    tmp_path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pot_path, tmp_path / "pot.mtp")
    pot = mlippy.mtp(str(tmp_path / "pot.mtp"))
    for atomic_number in atom_number_list:
        pot.add_atomic_type(atomic_number)

    calc = mlippy.MLIP_Calculator(pot, {})
    calc.use_cache = False
    return calc


def time_mlippy(
    pot_path: pathlib.Path,
    images: list[ase.atoms.Atoms],
):
    atom_number_list = []
    for n in images[0].get_atomic_numbers():
        if n not in atom_number_list:
            atom_number_list.append(n)
    calc = init_mlippy(pot_path, atom_number_list)
    # Make initial calc to not time things like compile time and things that are cachable
    calc.get_potential_energy(images[-1])
    with Timer(fmt.format("mlippy")):
        energies = [calc.get_potential_energy(_) for _ in images]
    return np.array(energies)


def time_mtp(
    pot_path: pathlib.Path,
    images: list[ase.atoms.Atoms],
    engine: str,
):
    parameters = read_mtp(pot_path)
    parameters["species"] = []
    for atomic_number in images[0].numbers:
        if atomic_number not in parameters["species"]:
            parameters["species"].append(atomic_number)
    calc = MTP(parameters, engine=engine)
    calc.use_cache = False

    # Make initial calc to not time things like compile time and things that are cachable
    with Timer(fmt.format(engine + " (0th)")):
        calc.get_potential_energy(images[-1])

    with Timer(fmt.format(engine)):
        energies = [calc.get_potential_energy(_) for _ in images]
    return np.array(energies)


def time_numpy(
    pot_path: pathlib.Path,
    images: list[ase.atoms.Atoms],
):
    return time_mtp(pot_path, images, "numpy")


def time_numba(
    pot_path: pathlib.Path,
    images: list[ase.atoms.Atoms],
):
    return time_mtp(pot_path, images, "numba")


def main() -> None:
    """Run benchmarks."""
    data_path = pathlib.Path(__file__).parent / "../tests/data_path"
    crystal = "cubic"
    for level in [6, 20]:
        for size_reps in [1, 3]:
            path = data_path / f"fitting/crystals/{crystal}/{level:02d}"
            cfg_path = path / "out.cfg"
            if not cfg_path.is_file():
                continue
            index = slice(0, 10)  # ":"
            images = read_cfg(cfg_path, index=index)
            images = [_.repeat(size_reps) for _ in images]
            number_of_atoms = len(images[0])
            print(
                f"\nTiming for {len(images)} images"
                f" of {number_of_atoms} atoms"
                f" with level {level}:"
            )
            pot_path = path / "pot.mtp"
            # To test with previous codes traj and pot
            # images = [
            #     ase.io.read("/Users/axelforslund/python_mtp/misc/md_atoms.traj")
            #     for _ in range(10)
            # ]
            # rng = np.random.default_rng()
            # [_.rattle(rng=rng) for _ in images]
            # pot_path = "/Users/axelforslund/direct-upsampling/directupsampling/tests/resources/Al_mtps/Al_fcc_pbe_10g.mtp"
            e_ref = time_mlippy(pot_path, images)
            if number_of_atoms < 300:
                e_numpy = time_numpy(pot_path, images)
                np.testing.assert_allclose(e_numpy, e_ref)
            e_numba = time_numba(pot_path, images)
            np.testing.assert_allclose(e_numba, e_ref)


if __name__ == "__main__":
    main()
