``motep grade``
===============

This command calculates the extrapolation grades for the configurations written
in ``data_in`` using ``potential_final`` and write them in ``data_out``.

Usage
-----

.. code-block:: bash

    motep grade motep.toml

``motep.toml``
--------------

.. code-block:: toml

    data_training = 'traning.cfg'  # original data for training
    data_in = 'in.cfg'  # data to be evaluated
    data_out = 'out.cfg'  # data with `MV_grade`
    potential_final = 'final.mtp'

    seed = 42  # random seed
    engine = 'cext'

    # grade
    algorithm = 'maxvol'  # {'maxvol', 'exaustive'}
