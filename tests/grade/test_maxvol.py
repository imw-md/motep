"""Tests for active-learning algorithms."""

import numpy as np

from motep.grade.maxvol import MaxVol


def test_maxvol() -> None:
    """Test if MaxVol algorithms give the same result.

    Since the MaxVol algorithm is a greedy algorithm and does not necessarily
    give the same result as the exhaustive search, but for simple systems they
    may agree.
    """
    rng = np.random.default_rng(42)
    matrix = rng.random((20, 10))
    indices_ref = MaxVol(algorithm="exhaust").run(matrix)
    indices = MaxVol(algorithm="maxvol", init_method="random", rng=rng).run(matrix)

    np.testing.assert_array_equal(indices, indices_ref)
