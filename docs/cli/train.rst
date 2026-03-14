``motep train``
===============

Usage
-----

.. code-block:: bash

    motep train motep.toml

or

.. code-block:: bash

    mpirun -np 4 motep train motep.toml

``motep.toml``
--------------

.. code-block:: toml

    data_training = 'training.cfg'
    potential_initial = 'initial.mtp'
    potential_final = 'final.mtp'

    # seed = 10  # random seed for initializing MTP parameters

    engine = 'cext'  # {'cext', 'numpy', 'numba', 'jax', 'mlippy'}

    [loss]  # setting for the loss function
    energy_weight = 1.0
    forces_weight = 0.01
    stress_weight = 0.001

    # optimization steps

    # style 1: simple
    # steps = ['Nelder-Mead', 'BFGS']

    # style 2: sophisticated
    # "optimized" specifies which parameters are optimized at the step.
    [[steps]]
    method = 'Nelder-Mead'
    optimized = ['species_coeffs', 'radial_coeffs', 'moment_coeffs']
    [steps.kwargs]
    tol = 1e-7
    [steps.kwargs.options]
    maxiter = 1000

    [[steps]]
    method = 'BFGS'
    optimized = ['species_coeffs', 'radial_coeffs', 'moment_coeffs']

If some of the following parameters are already given in ``initial.mtp``,
they are treated as the initial guess, which may or may not be optimized
depending on the above setting.

- ``scaling`` (*not* recommended to optimized)
- ``radial_coeffs``
- ``moment_coeffs``
- ``species_coeffs``
