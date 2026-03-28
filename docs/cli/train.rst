``motep train``
===============

This command trains the potential starting from ``potentials.initial`` based on
``configurations.training``.
The trained potential is written in ``potentials.final``.

Usage
-----

.. code-block:: bash

    motep train motep.train.toml

or

.. code-block:: bash

    mpirun -np 4 motep train motep.train.toml

``motep.train.toml``
--------------------

.. literalinclude:: motep.train.toml
    :language: toml

If some of the following parameters are already given in ``initial.mtp``,
they are treated as the initial guess, which may or may not be optimized
depending on the above setting.

- ``scaling`` (*not* recommended to optimized)
- ``radial_coeffs``
- ``moment_coeffs``
- ``species_coeffs``
