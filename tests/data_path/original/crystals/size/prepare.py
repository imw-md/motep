"""Script to obtain reference trajectories."""

import ase.io
import ase.units
import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.md import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from motep.io.mlip.cfg import write_cfg

for size in [1, 2]:
    atoms = bulk("Cu", cubic=True) * (size, size, size)
    MaxwellBoltzmannDistribution(
        atoms,
        temperature_K=1000.0,
        rng=np.random.default_rng(42),
    )
    atoms.calc = EMT()
    with VelocityVerlet(
        atoms,
        timestep=5.0 * ase.units.fs,
        trajectory=f"ase_{size}.traj",
    ) as md:
        md.run(50)
images = []
for size in [1, 2]:
    images.extend(ase.io.read(f"ase_{size}.traj", index=":"))
    write_cfg("training.cfg", images, key_energy="energy")
