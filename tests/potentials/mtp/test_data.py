"""Unit tests for MTPData and BasisData."""

import copy

import numpy as np
import pytest

from motep.potentials.mtp.data import BasisData, MTPData


def _make_mtp_data(
    species_count: int = 2,
    rfc: int = 4,
    rbs: int = 8,
    asm: int = 10,
) -> MTPData:
    return MTPData(
        species_count=species_count,
        radial_funcs_count=rfc,
        radial_basis=BasisData(type="RBChebyshev", min=2.0, max=5.0, size=rbs),
        alpha_scalar_moments=asm,
    )


class TestBasisData:
    def test_valid(self) -> None:
        bd = BasisData(type="RBChebyshev", min=2.0, max=5.0, size=8)
        assert bd.min == 2.0
        assert bd.max == 5.0
        assert bd.size == 8

    def test_min_ge_max_raises(self) -> None:
        with pytest.raises(ValueError, match="min"):
            BasisData(type="RBChebyshev", min=5.0, max=2.0, size=8)

    def test_negative_size_raises(self) -> None:
        with pytest.raises(ValueError, match="size"):
            BasisData(type="RBChebyshev", min=2.0, max=5.0, size=-1)

    def test_uninitialized_skips_validation(self) -> None:
        # size=0 and nan values should not raise
        BasisData()


class TestMTPDataInitialize:
    def test_shapes(self) -> None:
        spc, rfc, rbs, asm = 2, 4, 8, 10
        data = _make_mtp_data(spc, rfc, rbs, asm)
        rng = np.random.default_rng(0)
        data.initialize(rng)

        assert data.species_coeffs.shape == (spc,)
        assert data.moment_coeffs.shape == (asm,)
        assert data.radial_coeffs.shape == (spc, spc, rfc, rbs)

    def test_idempotent(self) -> None:
        """Calling initialize twice must not overwrite existing coefficients."""
        data = _make_mtp_data()
        rng = np.random.default_rng(0)
        data.initialize(rng)
        sc_before = data.species_coeffs.copy()
        data.initialize(rng)
        np.testing.assert_array_equal(data.species_coeffs, sc_before)


class TestMTPDataParameters:
    def test_roundtrip(self) -> None:
        data = _make_mtp_data()
        rng = np.random.default_rng(1)
        data.initialize(rng)
        params = data.parameters.copy()
        data2 = copy.deepcopy(data)
        data2.parameters = params
        np.testing.assert_array_equal(data2.parameters, params)

    def test_length_matches_number_of_parameters(self) -> None:
        data = _make_mtp_data()
        rng = np.random.default_rng(2)
        data.initialize(rng)
        assert len(data.parameters) == data.number_of_parameters_optimized

    def test_number_of_parameters_formula(self) -> None:
        spc, rfc, rbs, asm = 2, 4, 8, 10
        data = _make_mtp_data(spc, rfc, rbs, asm)
        rng = np.random.default_rng(3)
        data.initialize(rng)
        expected = asm + spc + spc * spc * rfc * rbs
        assert data.number_of_parameters_optimized == expected


class TestMTPDataGetBounds:
    def test_shape(self) -> None:
        data = _make_mtp_data()
        rng = np.random.default_rng(4)
        data.initialize(rng)
        bounds = data.get_bounds()
        assert bounds.shape == (data.number_of_parameters_optimized, 2)

    def test_all_finite_except_scaling(self) -> None:
        data = _make_mtp_data()
        rng = np.random.default_rng(5)
        data.initialize(rng)
        # Explicitly include scaling so we can test its (0, +inf) bound
        data.optimized = ["scaling", "moment_coeffs", "species_coeffs", "radial_coeffs"]
        bounds = data.get_bounds()
        # First row is scaling (0, +inf)
        assert bounds[0, 0] == 0.0
        assert np.isinf(bounds[0, 1])
        # All remaining bounds are unconstrained (-inf, +inf)
        assert np.all(np.isinf(bounds[1:]))
