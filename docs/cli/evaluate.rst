``motep evaluate``
==================

This command calculates energies, forces, and stresses for the configurations written
in ``data_in`` using ``potential_final`` and write them in ``data_out``.

Usage
-----

.. code-block:: bash

    motep evaluate motep.evaluate.toml

``motep.evaluate.toml``
-----------------------

.. code-block:: toml

    data_in = 'in.cfg'  # {'.cfg', '.xyz'}
    data_out = 'out.cfg'  # {'.cfg', '.xyz'}
    potential_final = 'final.mtp'
