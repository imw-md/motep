``motep evaluate``
==================

This command calculates energies, forces, and stresses for the configurations written
in ``configurations.initial`` using ``potentials.final``.
The evaluated energies, forces, stresses are written in ``configurations.final``.

Usage
-----

.. code-block:: bash

    motep evaluate motep.evaluate.toml

``motep.evaluate.toml``
-----------------------

.. literalinclude:: motep.evaluate.toml
    :language: toml
