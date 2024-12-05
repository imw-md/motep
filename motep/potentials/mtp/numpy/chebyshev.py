"""Radial basis functions based on Chebyshev polynomials."""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from motep.potentials.mtp.data import MTPData


class RadialBasisBase(ABC):
    """Base class of `RadialBasis`."""

    def __init__(self, mtp_data: MTPData) -> None:
        """Initialize `ChebyshevRadialBasis`."""
        self.mtp_data = mtp_data

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
    def calc_radial_part(
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
    basis_vs : npt.NDArray[np.float64]
        Values of the basis functions multiplied by the smoothing funciton
        with the shape of (radial_basis_size, neighbors).
        This is T(r) * (R_cut - r)^2 in Eq. (4) in [Podryabinkin_JCP_2023_MLIP]_.
    basis_ds : npt.NDArray[np.float64]
        Derivatives of the basis functions multiplied by the smoothing funciton
        with the shape of (radial_basis_size, neighbors).

    .. [Podryabinkin_JCP_2023_MLIP]
          E. Podryabinkin, K. Garifullin, A. Shapeev, and I. Novikov,
          J. Chem. Phys. 159, (2023).

    """

    def __init__(self, mtp_data: MTPData) -> None:
        """Initialize `ChebyshevRadialBasis`."""
        super().__init__(mtp_data)
        self.coeffs = None
        self.basis_vs = None
        self.basis_ds = None

    def vander(
        self,
        r_abs: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Calculate (pseudo) Vandermonde matrices.

        Parameters
        ----------
        r_abs : npt.NDArray[np.float64]
            Distances to the neighbors with the shape of (neighbors).

        Returns
        -------
        values : npt.NDArray[np.float64]
            (radial_basis_size, neighbors)
        derivs : npt.NDArray[np.float64]
            (radial_basis_size, neighbors)

        """
        min_dist = self.mtp_data.min_dist
        max_dist = self.mtp_data.max_dist
        radial_basis_size = self.mtp_data.radial_basis_size

        values = np.zeros((radial_basis_size, r_abs.size))
        derivs = np.zeros((radial_basis_size, r_abs.size))
        s = (2.0 * r_abs - (min_dist + max_dist)) / (max_dist - min_dist)
        d = 2.0 / (max_dist - min_dist)
        values[0] = 1.0
        values[1] = s
        derivs[0] = 0.0
        derivs[1] = d
        for i in range(2, radial_basis_size):
            values[i] = 2.0 * s * values[i - 1] - values[i - 2]
            derivs[i] = 2.0 * (d * values[i - 1] + s * derivs[i - 1]) - derivs[i - 2]

        return values, derivs

    def update_coeffs(self, coeffs: npt.NDArray[np.float64]) -> None:
        """Update radial basis coefficients."""
        self.coeffs = coeffs

    def calc_radial_part(
        self,
        r_abs: npt.NDArray[np.float64],
        itype: int,
        jtypes: list[int],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Calculate the rabial part.

        Returns
        -------
        radial_part_vs : npt.NDArray[np.float64]
            Values of the radial part (radial_funcs_count, neighbors).
            This corresponds to f_mu in Eq. (4) in [Podryabinkin_JCP_2023_MLIP]_.
        radial_part_ds : npt.NDArray[np.float64]
            Derivatives of the radial part (radial_funcs_count, neighbors).
            This corresponds to d f_mu / d r.

        .. [Podryabinkin_JCP_2023_MLIP]
          E. Podryabinkin, K. Garifullin, A. Shapeev, and I. Novikov,
          J. Chem. Phys. 159, (2023).

        """
        scaling = self.mtp_data.scaling
        max_dist = self.mtp_data.max_dist

        in_cutoff = r_abs < max_dist
        smooth_values = np.where(in_cutoff, scaling * (max_dist - r_abs) ** 2, 0.0)
        smooth_derivs = np.where(in_cutoff, -2.0 * scaling * (max_dist - r_abs), 0.0)

        vs0, ds0 = self.vander(r_abs)

        self.basis_vs = vs0 * smooth_values
        self.basis_ds = ds0 * smooth_values + vs0 * smooth_derivs

        radial_part_vs = np.sum(
            self.basis_vs.T[:, None, :] * self.coeffs[itype, jtypes],
            axis=-1,
        ).T
        radial_part_ds = np.sum(
            self.basis_ds.T[:, None, :] * self.coeffs[itype, jtypes],
            axis=-1,
        ).T

        return radial_part_vs, radial_part_ds


class ChebyshevPolynomialRadialBasis(RadialBasisBase):
    """Radial basis functions based on Chebyshev polynomials.

    Attributes
    ----------
    funcs : np.ndarray
        Array of `Chebyshev` objects.
    dfdrs : np.ndarray
        Array of `Chebyshev` objects for derivatives.

    """

    def __init__(self, mtp_data: MTPData) -> None:
        """Initialize `ChebyshevRadialBasis`."""
        super().__init__(mtp_data)
        self.funcs = None
        self.dfdrs = None

    def init_funcs(self, coeffs: npt.NDArray[np.float64]) -> None:
        """Initialize radial basis functions."""
        from numpy.polynomial import Chebyshev

        min_dist = self.mtp_data.min_dist
        max_dist = self.mtp_data.max_dist

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

    def calc_radial_part(
        self,
        r_abs: np.ndarray,
        itype: int,
        jtypes: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate values of radial basis functions."""
        scaling = self.mtp_data.scaling
        max_dist = self.mtp_data.max_dist
        radial_funcs_count = self.mtp_data.radial_funcs_count

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
