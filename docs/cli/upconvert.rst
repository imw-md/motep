``motep upconvert``
===================

This command up-converts an MTP potential with a higher level,
a larger radial basis size, and/or more species.
This enables us to retrain the potential with higher flexibility.

Usage
-----

.. code-block:: bash

    motep upconvert  # The default setting below is applied.

or

.. code-block:: bash

    motep upconvert motep.upconvert.toml

``motep.upconvert.toml``
------------------------

.. literalinclude:: motep.upconvert.toml
    :language: toml
