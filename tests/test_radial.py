"""Tests for `radial`."""

import numpy as np
import pytest

from motep.radial import (
    ChebyshevArrayRadialBasis,
    ChebyshevPolynomialRadialBasis,
    RadialBasisBase,
)

params = [
    (
        np.array(
            [
                [
                    [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]],
                ],
            ]
        ),
        np.array([[2.0, 2.0, 1.0]]),
        [0],
    ),
    (
        np.array(
            [
                [
                    [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]],
                    [[2.0, 3.0, 4.0], [3.0, 4.0, 5.0]],
                ],
                [
                    [[4.0, 5.0, 6.0], [5.0, 6.0, 7.0]],
                    [[6.0, 7.0, 8.0], [7.0, 8.0, 9.0]],
                ],
            ]
        ),
        np.array([[2.0, 2.0, 1.0], [1.0, 1.0, 2.0]]),
        [0, 1],
    ),
]


@pytest.mark.parametrize(("radial_coeffs", "r_ijs", "jtypes"), params)
@pytest.mark.parametrize(
    "radial_basis_class",
    [ChebyshevArrayRadialBasis, ChebyshevPolynomialRadialBasis],
)
def test_radial_funcs(
    radial_basis_class: RadialBasisBase,
    radial_coeffs: np.ndarray,
    r_ijs: np.ndarray,
    jtypes: list[int],
):
    mtp_data = {
        "scaling": 1.0,
        "min_dist": 2.0,
        "max_dist": 5.0,
        "radial_funcs_count": radial_coeffs.shape[2],
        "radial_basis_size": radial_coeffs.shape[3],
    }
    rb = radial_basis_class(mtp_data)
    rb.update_coeffs(radial_coeffs)
    r_abs = np.sqrt(np.add.reduce(r_ijs**2, axis=1))
    _, radial_basis_derivs = rb.calculate(r_abs, 0, jtypes=jtypes)
    radial_basis_derivs = radial_basis_derivs[:, :, None] * r_ijs / r_abs[:, None]

    dx = 1e-6

    r_ijs_p = r_ijs + [[dx, 0.0, 0.0]]
    r_abs_p = np.sqrt(np.add.reduce(r_ijs_p**2, axis=1))
    radial_basis_values_p, _ = rb.calculate(r_abs_p, 0, jtypes=jtypes)

    r_ijs_m = r_ijs - [[dx, 0.0, 0.0]]
    r_abs_m = np.sqrt(np.add.reduce(r_ijs_m**2, axis=1))
    radial_basis_values_m, _ = rb.calculate(r_abs_m, 0, jtypes=jtypes)

    radial_basis_derivs_fd = radial_basis_values_p - radial_basis_values_m
    radial_basis_derivs_fd /= 2.0 * dx
    np.testing.assert_allclose(
        radial_basis_derivs[0, 0, 0],
        radial_basis_derivs_fd[0, 0],
    )
