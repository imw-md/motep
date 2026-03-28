``motep grade``
===============

This command calculates the extrapolation grades for the configurations written
in ``configurations.initial`` using ``potentials.final`` and ``configurations.training``.
The evaluated extrapolation grades are written in ``configurations.final``.

Usage
-----

.. code-block:: bash

    motep grade motep.grade.toml

``motep.grade.toml``
--------------------

.. literalinclude:: motep.grade.toml
    :language: toml
