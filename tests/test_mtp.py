"""Tests for PyMTP."""

import pathlib

import numpy as np
import pytest

from motep.io.mlip.cfg import read_cfg
from motep.io.mlip.mtp import read_mtp
from motep.mtp import MTP


@pytest.mark.parametrize("level", [2, 4, 6, 8, 10])
@pytest.mark.parametrize(
    ("molecule", "species"),
    [[762, {1: 0}], [291, {6: 0, 1: 2}]],
)
# @pytest.mark.parametrize("molecule", [762])
def test_mtp(
    molecule: int,
    species: dict[int, int],
    level: int,
    data_path: pathlib.Path,
) -> None:
    """Test PyMTP."""
    if molecule == 291 and level == 4:
        pytest.skip()
    path = data_path / f"fitting/{molecule}/{level:02d}"
    parameters = read_mtp(path / "pot.mtp")
    # parameters["species"] = species
    mtp = MTP(parameters)
    images = [read_cfg(path / "out.cfg", index=0)]
    mtp._initiate_neighbor_list(images[0])

    energies_ref = np.array([_.get_potential_energy() for _ in images])
    energies = np.array([mtp.get_energy(_) for _ in images]).reshape(-1)
    print(np.array(energies), np.array(energies_ref))
    np.testing.assert_allclose(energies, energies_ref)
