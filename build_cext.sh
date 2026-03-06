#!/bin/bash
# Convenience build script for the C extension meant for development and
# testing. For production builds, rather use something like `pip install .`.

set -e

echo "Building MTP C extension..."

# Clean previous builds
rm -rf build/ dist/ *.egg-info

# Build the extension
python setup.py build_ext --inplace

echo "Build complete!"
