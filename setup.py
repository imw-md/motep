"""Setup script for building C extensions."""

import numpy as np
from setuptools import Extension, setup

# Define the MTP C extension
mtp_cext = Extension(
    "motep.potentials.mtp.cext._mtp_cext",
    sources=[
        "motep/potentials/mtp/cext/mtp_cext.c",
        "motep/potentials/mtp/cext/mtp_cext_module.c",
    ],
    include_dirs=[
        np.get_include(),
        "motep/potentials/mtp/cext",
    ],
    extra_compile_args=["-O3", "-ffast-math", "-march=native"],
)

if __name__ == "__main__":
    setup(ext_modules=[mtp_cext])
