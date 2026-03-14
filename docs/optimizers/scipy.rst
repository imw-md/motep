SciPy optimizers
================

.. |scipy-minimize| replace:: ``scipy.optimize.minimize``
.. _scipy-minimize: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

.. |scipy-DA| replace:: ``scipy.optimize.dual_annealing``
.. _scipy-DA: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html

.. |scipy-DE| replace:: ``scipy.optimize.differential_evolution``
.. _scipy-DE: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html

The gradient-based local optimizers and the global optimizers in SciPy are available to
determine MTP parameters.

Local optimizers
----------------

We can give the method name for |scipy-minimize|_.

For CLI (``motep.toml`` for ``motep train``):

.. code-block:: toml

    [[steps]]
    method = "BFGS"
    optimized = ["species_coeffs", "radial_coeffs", "moment_coeffs"]

For Python API:

.. code-block:: python

    from motep.trainer import Trainer

    method = "BFGS"
    optimized = ["species_coeffs", "radial_coeffs", "moment_coeffs"]
    Trainer(..., steps=[{"method": method, "optimized": optimized}])

Methods like ``BGFS`` and ``Nelder-Mead`` can be specified.
Optimizers with constraints such as ``L-BFGS-B`` are also available,
but since the fixed parameters are handled on the MOTEP side,
they are not particularly recommended.

Global optimizers
-----------------

SciPy global optimizers can be specified.

- |scipy-DA|_: ``DA``
- |scipy-DE|_: ``DE``
