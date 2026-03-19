"""Tests for active-learning algorithms."""

from pathlib import Path

import numpy as np

from motep.grader.grader import Grader
from motep.io.mlip.cfg import read_cfg
from motep.io.mlip.mtp import read_mtp


def test_maxvol(data_path: Path) -> None:
    """Test if MaxVol algorithm gives the same results as the exhaustive search.

    Since the MaxVol algorithm is a greedy algorithm and does not necessarily
    give the same result as the exhaustive search, but for simple systems they
    may agree.
    """
    path = data_path / "fitting/crystals/multi/06"
    images_training = read_cfg(path / "out.cfg", index="::25")
    mtp_data = read_mtp(path / "pot.mtp")
    rng = np.random.default_rng(42)

    grader_ref = Grader(mtp_data, algorithm="exhaustive", rng=rng)
    grader_ref.update(images_training)
    det_ref = np.linalg.det(grader_ref.active_set_matrix)

    grader = Grader(mtp_data, algorithm="maxvol", rng=rng)
    grader.update(images_training)
    det = np.linalg.det(grader.active_set_matrix)

    np.testing.assert_array_equal(grader_ref.indices, grader.indices)
    np.testing.assert_almost_equal(det_ref, det)
