# Installation

## GitHub

The development version is available from [GitHub](https://github.com/imw-md/motep).

```bash
pip install git+https://github.com/imw-md/motep.git
```

For development, you can first clone the GitHub repository and install the package in the editable mode.

Note that, for the editable installation with C extensions, [``--no-build-isolation``](https://pip.pypa.io/en/stable/cli/pip_install/#cmdoption-no-build-isolation) needs to be specified.

```bash
git clone git@github.com:imw-md/motep.git
cd motep
pip install --no-build-isolation -e .
```
