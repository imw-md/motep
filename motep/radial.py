"""Radial basis functions based on Chebyshev polynomials."""

from typing import Any

import numpy as np
import numpy.typing as npt


class ChebyshevPolynomialRadialBasis:
    """Radial basis functions based on Chebyshev polynomials.

    Attributes
    ----------
    funcs : np.ndarray
        Array of `Chebyshev` objects.
    dfdrs : np.ndarray
        Array of `Chebyshev` objects for derivatives.

    """

    def __init__(self, dict_mtp: dict[str, Any]) -> None:
        """Initialize `ChebyshevRadialBasis`."""
        self.dict_mtp = dict_mtp
        self.funcs = None
        self.dfdrs = None

    def init_funcs(
        self,
        radial_coeffs: npt.NDArray[np.float64],
    ) -> None:
        """Initialize radial basis functions."""
        from numpy.polynomial import Chebyshev

        min_dist = self.dict_mtp["min_dist"]
        max_dist = self.dict_mtp["max_dist"]

        radial_basis_funcs = []
        radial_basis_dfdrs = []  # derivatives
        domain = [min_dist, max_dist]
        nspecies, _, nmu, _ = radial_coeffs.shape
        for i0 in range(nspecies):
            for i1 in range(nspecies):
                for i2 in range(nmu):
                    p = Chebyshev(radial_coeffs[i0, i1, i2], domain=domain)
                    radial_basis_funcs.append(p)
                    radial_basis_dfdrs.append(p.deriv())
        shape = nspecies, nspecies, nmu
        self.funcs = np.array(radial_basis_funcs).reshape(shape)
        self.dfdrs = np.array(radial_basis_dfdrs).reshape(shape)

    def update_coeffs(self, radial_coeffs: npt.NDArray[np.float64]) -> None:
        """Update radial basis coefficients."""
        nspecies, _, nmu, _ = radial_coeffs.shape
        for i0 in range(nspecies):
            for i1 in range(nspecies):
                for i2 in range(nmu):
                    p = self.funcs[i0, i1, i2]
                    p.coef = radial_coeffs[i0, i1, i2]
                    self.dfdrs[i0, i1, i2] = p.deriv()

    def calculate(
        self,
        r_abs: np.ndarray,
        itype: int,
        jtypes: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate values of radial basis functions."""
        scaling = self.dict_mtp["scaling"]
        max_dist = self.dict_mtp["max_dist"]
        radial_funcs_count = self.dict_mtp["radial_funcs_count"]

        is_within_cutoff = r_abs < max_dist
        smooth_values = scaling * (max_dist - r_abs) ** 2
        smooth_derivs = -2.0 * scaling * (max_dist - r_abs)
        rb_values = np.zeros((radial_funcs_count, r_abs.size))
        rb_derivs = np.zeros((radial_funcs_count, r_abs.size))
        for mu in range(radial_funcs_count):
            for j, jtype in enumerate(jtypes):
                if is_within_cutoff[j]:
                    rb_func = self.funcs[itype, jtype, mu]
                    rb_dfdr = self.dfdrs[itype, jtype, mu]
                    v = rb_func(r_abs[j]) * smooth_values[j]
                    rb_values[mu, j] = v
                    d0 = rb_dfdr(r_abs[j]) * smooth_values[j]
                    d1 = rb_func(r_abs[j]) * smooth_derivs[j]
                    d = d0 + d1
                    rb_derivs[mu, j] = d

        return rb_values, rb_derivs
