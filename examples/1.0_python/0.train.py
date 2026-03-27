"""
Trainer
=======
"""

# %%
# This example shows how to use the :class:`~motep.train.Trainer` class to train the potential.
import logging

import ase.io
from ase.visualize.plot import plot_atoms

from motep.io.mlip.mtp import read_mtp, write_mtp
from motep.train import Trainer

# %%
# If you need a log, you can set the following.

logging.basicConfig(level=logging.INFO, format="%(message)s")

# %%
# We first load the atomic configuration to compute.
images = ase.io.read("../0.0_train/ase.xyz", index=":")
ax = plot_atoms(images[0])

# %%
# We next load the initial (likely untrained) potential.
mtp_data = read_mtp("initial.mtp")
mtp_data.species = [6, 1]
mtp_data

# %%
# We then make the :class:`~motep.train.Trainer` class and train the potential by the images.
trainer = Trainer(mtp_data, seed=42)
loss = trainer.train(images)

# %%
# We can see the final loss function value.
loss(mtp_data.parameters)

# %%
# After the training, the given `mtp_data` is updated in-place.
# We finally store the trained potential in a new file.
write_mtp("final.mtp", mtp_data)
