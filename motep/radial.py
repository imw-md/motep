"""Radial basis functions based on Chebyshev polynomials."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt


class RadialBasisBase(ABC):
    """Base class of `RadialBasis`."""

    def __init__(self, dict_mtp: dict[str, Any]) -> None:
        """Initialize `ChebyshevRadialBasis`."""
        self.dict_mtp = dict_mtp

    @abstractmethod
    def update_coeffs(self, coeffs: npt.NDArray[np.float64]) -> None:
        """Update radial basis coefficients.

        Parameters
        ----------
        coeffs : npt.NDArray[np.float64]
            Coefficients of radial basis functions with the shape of
            (species_count, species_count, radial_funcs_count, radial_basis_size).

        """

    @abstractmethod
    def calculate(
        self,
        r_abs: npt.NDArray[np.float64],
        itype: int,
        jtypes: list[int],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Calculate values of radial basis functions."""


class ChebyshevArrayRadialBasis(RadialBasisBase):
    """Radial basis functions based on Chebyshev polynomials.

    Attributes
    ----------
    values0 : npt.NDArray[np.float64]
        Values of the (pseudo) Vandermonde matrix multiplied with the smoothing
        funciton with the shape of (radial_funcs_count, radial_basis_size).

    """

    def __init__(self, dict_mtp: dict[str, Any]) -> None:
        """Initialize `ChebyshevRadialBasis`."""
        super().__init__(dict_mtp)
        self.coeffs = None
        self.values0 = None
        self.derivs0 = None

    def vander(
        self,
        r_abs: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Calculate (pseudo) Vandermonde matrices.

        Returns
        -------
        values : npt.NDArray[np.float64]
            (neighbors, radial_basis_size)

        """
        min_dist = self.dict_mtp["min_dist"]
        max_dist = self.dict_mtp["max_dist"]
        radial_basis_size = self.dict_mtp["radial_basis_size"]

        values = np.zeros((radial_basis_size, r_abs.size))
        derivs = np.zeros((radial_basis_size, r_abs.size))
        s = (2.0 * r_abs - (min_dist + max_dist)) / (max_dist - min_dist)
        d = 2.0 / (max_dist - min_dist)
        values[0] = 1.0
        values[1] = s
        derivs[0] = 0.0
        derivs[1] = 2.0 / (max_dist - min_dist)
        for i in range(2, radial_basis_size):
            values[i] = 2.0 * s * values[i - 1] - values[i - 2]
            derivs[i] = 2.0 * (d * values[i - 1] + s * derivs[i - 1]) - derivs[i - 2]

        return values.T, derivs.T

    def update_coeffs(self, coeffs: npt.NDArray[np.float64]) -> None:
        """Update radial basis coefficients."""
        self.coeffs = coeffs

    def calculate(
        self,
        r_abs: npt.NDArray[np.float64],
        itype: int,
        jtypes: list[int],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Calculate rabial basis values.

        Returns
        -------
        rb_values : npt.NDArray[np.float64]
            Values of radial basis functions (radial_funcs_count, neighbors).
        rb_derivs : npt.NDArray[np.float64]
            Derivatives of radial basis functions (radial_funcs_count, neighbors).

        """
        scaling = self.dict_mtp["scaling"]
        max_dist = self.dict_mtp["max_dist"]

        in_cutoff = r_abs < max_dist
        smooth_values = np.where(in_cutoff, scaling * (max_dist - r_abs) ** 2, 0.0)
        smooth_derivs = np.where(in_cutoff, -2.0 * scaling * (max_dist - r_abs), 0.0)
        vs0, ds0 = self.vander(r_abs)
        self.values0 = vs0 * smooth_values[:, None]
        self.derivs0 = ds0 * smooth_values[:, None] + vs0 * smooth_derivs[:, None]
        tmp0 = np.sum(vs0[:, None, :] * self.coeffs[itype, jtypes], axis=-1).T
        tmp1 = np.sum(ds0[:, None, :] * self.coeffs[itype, jtypes], axis=-1).T
        rb_values = tmp0 * smooth_values
        rb_derivs = tmp1 * smooth_values + tmp0 * smooth_derivs

        return rb_values, rb_derivs


class ChebyshevPolynomialRadialBasis(RadialBasisBase):
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
        super().__init__(dict_mtp)
        self.funcs = None
        self.dfdrs = None

    def init_funcs(self, coeffs: npt.NDArray[np.float64]) -> None:
        """Initialize radial basis functions."""
        from numpy.polynomial import Chebyshev

        min_dist = self.dict_mtp["min_dist"]
        max_dist = self.dict_mtp["max_dist"]

        radial_basis_funcs = []
        radial_basis_dfdrs = []  # derivatives
        domain = [min_dist, max_dist]
        nspecies, _, nmu, _ = coeffs.shape
        for i0 in range(nspecies):
            for i1 in range(nspecies):
                for i2 in range(nmu):
                    p = Chebyshev(coeffs[i0, i1, i2], domain=domain)
                    radial_basis_funcs.append(p)
                    radial_basis_dfdrs.append(p.deriv())
        shape = nspecies, nspecies, nmu
        self.funcs = np.array(radial_basis_funcs).reshape(shape)
        self.dfdrs = np.array(radial_basis_dfdrs).reshape(shape)

    def update_coeffs(self, coeffs: npt.NDArray[np.float64]) -> None:
        """Update radial basis coefficients."""
        if self.funcs is None:
            self.init_funcs(coeffs)
            return
        nspecies, _, nmu, _ = coeffs.shape
        for i0 in range(nspecies):
            for i1 in range(nspecies):
                for i2 in range(nmu):
                    p = self.funcs[i0, i1, i2]
                    p.coef = coeffs[i0, i1, i2]
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
