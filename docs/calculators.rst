ASE calculators
===============

Moment tensor potentials
------------------------

.. code-block:: python

    from motep.calculator import MTP

    calc = MTP(..., engine="cext")

There are multiple backend implementations, which can be specified by ``engine``.

1. ``cext``: `C <https://www.c-language.org/>`_ implementation

This is the fastest and the recommended implementation.

2. ``numpy``: `NumPy <https://numpy.org/>`_ implementation

This is slow but does not require other packages.

3. ``numba``: `Numba <https://numba.pydata.org/>`_ implementation

This is faster than ``numpy`` but slower than ``cext``.

4. ``jax``: `JAX <https://docs.jax.dev/>`_ implementation

5. ``mlippy``: Wrapper of the Python/Cython implementation in `mlip-2 <https://gitlab.com/ashapeev/mlip-2>`_

This is faster than ``numba`` for evaluation but has several limitations, particularly for training.

1. Our modified version of |mlippy|_ needs to be installed.
2. This requires file IO for the potential file at every step of training in ``motep``.
3. The training with analytical gradients is not available in ``motep``.
4. The extrapolation grade is not available.
5. ``mlippy`` seems not publicly supported anymore by the ``mlip-2`` developers.
6. The ``mlippy`` engine will be removed from ``motep`` in the near future.

.. |mlippy| replace:: ``mlippy``
.. _mlippy: https://gitlab.com/yuzie007/mlip-2/-/tree/mlippy
