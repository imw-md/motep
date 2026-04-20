"""Unit tests for MagMTPData."""

import copy

import numpy as np
import pytest

from motep.potentials.mmtp.data import MagMTPData
from motep.potentials.mtp.data import BasisData, MTPData


def _make_mag_mtp_data(
    species_count: int = 2,
    rfc: int = 4,
    rbs: int = 8,
    mbs: int = 2,
    asm: int = 10,
) -> MagMTPData:
    return MagMTPData(
        species_count=species_count,
        radial_funcs_count=rfc,
        radial_basis=BasisData(type="RBChebyshev", min=2.0, max=5.0, size=rbs),
        magnetic_basis=BasisData(type="BChebyshev", min=-3.5, max=3.5, size=mbs),
        alpha_scalar_moments=asm,
    )


class TestMagMTPDataInitialize:
    def test_shapes(self) -> None:
        spc, rfc, rbs, mbs, asm = 2, 4, 8, 2, 10
        data = _make_mag_mtp_data(spc, rfc, rbs, mbs, asm)
        rng = np.random.default_rng(0)
        data.initialize(rng)

        assert data.species_coeffs.shape == (spc,)
        assert data.moment_coeffs.shape == (asm,)
        assert data.radial_coeffs.shape == (spc, spc, rfc, rbs * mbs**2)

    def test_idempotent(self) -> None:
        """Calling initialize twice must not overwrite existing coefficients."""
        data = _make_mag_mtp_data()
        rng = np.random.default_rng(2)
        data.initialize(rng)
        sc_before = data.species_coeffs.copy()
        data.initialize(rng)
        np.testing.assert_array_equal(data.species_coeffs, sc_before)


class TestMagMTPDataFromBase:
    def test_returns_mag_mtp_data(self) -> None:
        base = MTPData(species_count=2, radial_funcs_count=4, alpha_scalar_moments=10)
        result = MagMTPData.from_base(base)
        assert isinstance(result, MagMTPData)

    def test_preserves_base_fields(self) -> None:
        base = MTPData(
            species_count=3,
            radial_funcs_count=6,
            alpha_scalar_moments=15,
            scaling=2.5,
        )
        result = MagMTPData.from_base(base)
        assert result.species_count == base.species_count
        assert result.radial_funcs_count == base.radial_funcs_count
        assert result.alpha_scalar_moments == base.alpha_scalar_moments
        assert result.scaling == base.scaling

    def test_has_magnetic_basis(self) -> None:
        base = MTPData()
        result = MagMTPData.from_base(base)
        assert isinstance(result.magnetic_basis, BasisData)


class TestMagMTPDataParameters:
    def test_roundtrip(self) -> None:
        data = _make_mag_mtp_data()
        rng = np.random.default_rng(3)
        data.initialize(rng)
        params = data.parameters.copy()
        data2 = copy.deepcopy(data)
        data2.parameters = params
        np.testing.assert_array_equal(data2.parameters, params)

    def test_length_matches_number_of_parameters(self) -> None:
        data = _make_mag_mtp_data()
        rng = np.random.default_rng(4)
        data.initialize(rng)
        assert len(data.parameters) == data.number_of_parameters_optimized

    def test_number_of_parameters_formula(self) -> None:
        spc, rfc, rbs, mbs, asm = 2, 4, 8, 2, 10
        data = _make_mag_mtp_data(spc, rfc, rbs, mbs, asm)
        rng = np.random.default_rng(5)
        data.initialize(rng)
        nrb = rbs * mbs**2
        expected = asm + spc + spc * spc * rfc * nrb
        assert data.number_of_parameters_optimized == expected
