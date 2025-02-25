"""Tests for active-learning algorithms."""

from pathlib import Path

import numpy as np

from motep.active.algorithms import ExhaustiveAlgorithm, MaxVolAlgorithm
from motep.io.mlip.cfg import read_cfg
from motep.io.mlip.mtp import read_mtp


def test_maxvol(data_path: Path) -> None:
    """Test if the MaxVol algorithm gives the same results as the exhaustive search."""
    path = data_path / "fitting/crystals/multi/06"
    images_training = read_cfg(path / "out.cfg", index="::25")
    mtp_data = read_mtp(path / "pot.mtp")
    rng = np.random.default_rng(42)
    algo_ref = ExhaustiveAlgorithm(images_training, mtp_data, engine="numba", rng=rng)
    det_ref = np.linalg.det(algo_ref.matrix[algo_ref.indices])
    algo = MaxVolAlgorithm(images_training, mtp_data, engine="numba", rng=rng)
    det = np.linalg.det(algo.matrix[algo.indices])
    np.testing.assert_array_equal(algo_ref.indices, algo.indices)
    np.testing.assert_almost_equal(det_ref, det)
