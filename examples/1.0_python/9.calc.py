"""
ASE calculator
==============
"""
# %%
# This example shows how to use the trained potential in Python as an ASE calculator.
# Suppose you finished `0.0_train`, and we then write the following `run.py`.

import ase.io
from ase.visualize.plot import plot_atoms

from motep.calculator import MTP
from motep.io.mlip.mtp import read_mtp

# %%
# We first load the atomic configuration to compute.
atoms = ase.io.read("../0.0_train/ase.xyz")
ax = plot_atoms(atoms)

# %%
# We next load the trained potential.
potential = read_mtp("final.mtp")  # copied from ../0.0_train
potential.species = [6, 1]

# %%
# We then assign the potential to `atoms` and trigger the calculation.
atoms.calc = MTP(potential)
atoms.get_potential_energy()

# %%
# We finally store the energy and the forces in a new file.
atoms.write("final.xyz")

# %%
# Run (local)
# -----------
#
# .. code-block:: console
#
#     python run.py
#
# Run (slurm)
# -----------
#
# You can also submit the above job to a computational node using SLURM.
#
# The batch script `script_001_1h.sh` is in the `slurm` directory.
#
# .. code-block:: bash
#
#     #!/usr/bin/env bash
#     #SBATCH -J script
#     #SBATCH --time 1:00:00
#     #SBATCH --export=HOME
#
#     source $HOME/.bashrc
#
#     # https://www.gnu.org/software/bash/manual/bash.html#Special-Parameters
#     # ($@) Expands to the positional parameters, starting from one.
#     $@
#
# For later convenience, we put the script in close to the home directory.
#
# .. code-block:: console
#
#     mkdir -p ~/slurm
#     cp ../slurm/script_001_1h.sh ~/slurm
#
# We can submit the above python script as
#
# .. code-block:: console
#
#     sbatch ~/slurm/script_001_1h.sh python run.py
#
