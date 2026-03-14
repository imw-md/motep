Installation
============

GitHub
------

The development version is available from `GitHub <https://github.com/imw-md/motep>`_.

.. code-block:: bash

    pip install git+https://github.com/imw-md/motep.git

For development:

- Clone the GitHub repository.
- Install build-time requirements.
- Install ``motep`` in the editable mode with |no-build-isolation|_.

.. code-block:: bash

    git clone git@github.com:imw-md/motep.git
    cd motep
    pip install meson-python setuptools_scm "numpy>=2,<3"
    pip install --no-build-isolation -e .

.. |no-build-isolation| replace:: ``--no-build-isolation``
.. _no-build-isolation: https://pip.pypa.io/en/stable/cli/pip_install/#cmdoption-no-build-isolation
