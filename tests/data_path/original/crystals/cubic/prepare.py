"""Script to obtain reference trajectories."""

import ase.io
import ase.units
import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.md import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from motep.io.mlip.cfg import write_cfg

atoms = bulk("Cu", cubic=True) * (2, 2, 2)
MaxwellBoltzmannDistribution(
    atoms,
    temperature_K=1000.0,
    rng=np.random.default_rng(42),
)
atoms.calc = EMT()
with VelocityVerlet(
    atoms,
    timestep=5.0 * ase.units.fs,
    trajectory="ase.traj",
) as md:
    md.run(1000)
images = ase.io.read("ase.traj", index=":")
write_cfg("training.cfg", images, key_energy="energy")
