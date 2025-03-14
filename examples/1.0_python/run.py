"""Run."""

import ase.io

from motep.calculator import MTP
from motep.io.mlip.mtp import read_mtp

atoms = ase.io.read("../0.0_train/ase.xyz")
potential = read_mtp("../0.0_train/final.mtp")
potential.species = [6, 1]
atoms.calc = MTP(potential)
atoms.get_potential_energy()
atoms.write("final.xyz")
