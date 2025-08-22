import pytest

from motep.potentials.mtp.jax.moment import MomentBasis


@pytest.mark.parametrize(
    "max_level",
    [2, 4, 6, 8, 10, 12, 14, 16],  # , 18, 20],
)
def test_moment_contractions(max_level) -> None:
    mb = MomentBasis(max_level)

    moments_ref = mb.read_moments()

    moments = mb.find_moment_contractions()

    assert moments == moments_ref
