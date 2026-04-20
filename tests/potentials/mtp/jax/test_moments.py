import pytest

from motep.potentials.mtp.jax.moment import MomentBasis, _flatten_to_moments


@pytest.mark.parametrize(
    "max_level",
    [2, 4, 6, 8, 10, 12, 14, 16],  # , 18, 20],
)
def test_moment_contractions(max_level) -> None:
    mb = MomentBasis(max_level)

    moments_ref = mb.read_moments()

    moments = mb.find_moment_contractions()

    print(moments_ref)
    print(moments)
    assert len(moments) == len(moments_ref)
    assert all([_ in moments_ref for _ in moments])
    assert moments == moments_ref


# @pytest.mark.parametrize("max_level", [2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
@pytest.mark.parametrize("max_level", [16, 18])
@pytest.mark.parametrize("max_contraction_length", [1, 2, 3, 4, 5, 6])  # Default 4
@pytest.mark.parametrize("max_mu", [5])  # Default 5
@pytest.mark.parametrize("max_nu", [10])  # Default 10
def test_nondefault_parameters(
    max_level: int,
    max_contraction_length: int,
    max_mu: int,
    max_nu: int,
) -> None:
    """Test `find_moment_contractions`.

    This test is designed for higher levels, where the length is actually
    limited by `max_contraction_length` not the level. At the moment, only
    `max_contraction_length` is varied.

    """
    mb = MomentBasis(
        max_level,
        max_contraction_length,
        max_mu,
        max_nu,
    )

    moments = mb.find_moment_contractions()
    assert max(len(_flatten_to_moments(_)) for _ in moments) == max_contraction_length
