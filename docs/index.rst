MOTEP: Moment tensor potentials in Python
=========================================

MOTEP is a Python implementation of moment tensor potentials (MTPs) [1]_.

- Both the :doc:`CLI <cli/index>` and the Python API, including the :doc:`ASE calculator <calculators>`, are available.
- :doc:`Various optimizers can be applied in various orders to train MTP parameters. <optimizers/index>`
- :doc:`The loss function can be flexibly customized. <loss>`
- :doc:`File IO for the MLIP-code formats are offered. <io>`

Please cite the paper [2]_ when publishing the results obtained with MOTEP.

.. toctree::
    :maxdepth: 1

    installation
    cli/index
    calculators
    optimizers/index
    loss
    io
    changelog
    authors

.. [1]
    A. V. Shapeev, Multiscale Model. Simul. 14, 1153 (2016).
    https://doi.org/10.1137/15m1054183

.. [2]
    Y. Ikeda, A. Forslund, P. Kumar, Y. Ou, J. H. Jung, A. Köhn, and B. Grabowski, J. Chem. Theory Comput. (2026).
    https://doi.org/10.1021/acs.jctc.5c02045
