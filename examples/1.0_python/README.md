# 1.0_python

This example shows how to use the trained potential in Python as an ASE calculator.
Suppose you finished `0.0_train`, and we then write the following `run.py`.
```python
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
```
## Run (local)

```console
$ python run.py
```

## Run (slurm)

You can also submit the above job to a computational node using SLURM.

The batch script `script_001_1h.sh` is in the `slurm` directory.

```bash
#!/usr/bin/env bash
#SBATCH -J script
#SBATCH --time 1:00:00
#SBATCH --export=HOME

source $HOME/.bashrc

# https://www.gnu.org/software/bash/manual/bash.html#Special-Parameters
# ($@) Expands to the positional parameters, starting from one.
$@
```

For later convenience, we put the script in close to the home directory.

```console
mkdir -p ~/slurm
cp ../slurm/script_001_1h.sh ~/slurm
```

We can submit the above python script as

```console
sbatch ~/slurm/script_001_1h.sh python run.py
```
