"""Module for moment tensors."""

from warnings import warn

import numpy as np
import numpy.typing as npt

from motep.potentials.mtp.data import MTPData

from .moment import MomentBasis
from .utils import TEST_R_UNITS, TEST_RB_VALUES, make_tensor

# dict mapping MLIP moments count to level, used for conversion
moments_count_to_level_map = {
    1: 2,
    2: 4,
    8: 6,
    18: 8,
    41: 10,
    84: 12,
    174: 14,
    350: 16,
    718: 18,
    1352: 20,
    2621: 22,
    4991: 24,
    9396: 26,
    17366: 28,
}


class MLIPMomentBasis:
    """Simplified verison of numpy engine `MomentBasis`."""

    def __init__(self, mtp_data: MTPData) -> None:
        """Initialize `MomentBasis`."""
        self.mtp_data = mtp_data

    def calculate(
        self,
        r_ijs_unit: npt.NDArray[np.float64],  # (neighbors, 3)
        rb_values: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        r"""Calculate basis functions and their derivatives.

        Parameters
        ----------
        r_ijs : np.ndarray (number_of_neighbors, 3)
            :math:`\mathbf{r}_j - \mathbf{r}_i`,
            where i is the center atom, and j are the neighboring atoms.
        rb_values : np.ndarray (max_mu, number_of_neighbors)

        Returns
        -------
        basis_vals : np.ndarray (alpha_moments_count)
            Values of the basis functions.

        """
        amc = self.mtp_data.alpha_moments_count
        alpha_index_basic = self.mtp_data.alpha_index_basic
        alpha_index_times = self.mtp_data.alpha_index_times
        alpha_moment_mapping = self.mtp_data.alpha_moment_mapping

        moment_values = np.zeros(amc)

        # Precompute powers
        max_pow = np.max(alpha_index_basic)
        r_unit_pows = _calc_r_unit_pows(r_ijs_unit, max_pow + 1)

        # Compute basic moments
        mu, xpow, ypow, zpow = alpha_index_basic.T

        # `mult0.shape == (alpha_index_basic_count, neighbors)`
        mult0 = (
            r_unit_pows[xpow, :, 0] * r_unit_pows[ypow, :, 1] * r_unit_pows[zpow, :, 2]
        )

        # f * tensor = dMb/dc (before summation over neighbors)
        # `val.shape == (alpha_index_basis_count, radial_basis_size, neighbors)`
        val = rb_values[mu, :] * mult0[:, :]
        moment_values[: mu.size] = val.sum(axis=1)
        _contract_moments(moment_values, alpha_index_times)

        return moment_values[alpha_moment_mapping]


def _calc_r_unit_pows(r_unit: np.ndarray, max_pow: int) -> np.ndarray:
    r_unit_pows = np.empty((max_pow, *r_unit.shape))
    r_unit_pows[0] = 1.0
    r_unit_pows[1:] = r_unit
    np.multiply.accumulate(r_unit_pows[1:], out=r_unit_pows[1:])
    return r_unit_pows


def _contract_moments(
    moment_values: npt.NDArray[np.float64],
    alpha_index_times: npt.NDArray[np.int64],
) -> None:
    """Compute contractions of moments."""
    for ait in alpha_index_times:
        i1, i2, mult, i3 = ait
        moment_values[i3] += mult * moment_values[i1] * moment_values[i2]


class BasisConverter:
    """Class to store and convert mapping between MTP basis functions and coefficients."""

    def __init__(self, moment_basis: MomentBasis):
        self.moment_basis = moment_basis
        self.remapped_coeffs = None

    def remap_mlip_moment_coeffs(self, mtp_data: MTPData):
        """Perform a remapping of the MLIP coeffs loaded to this potentials basis.

        This might be needed because the ordereing might be different or some basis elements omitted.
        """
        r_unit = TEST_R_UNITS
        rb_values = TEST_RB_VALUES

        # Calculate the MLIP like basis for test vectors (rb_values are the same)
        # Store coeffs as values with mlip basis values as keys
        bc_map = {}
        mlip_moment_basis = MLIPMomentBasis(mtp_data)
        mlip_basis_values = mlip_moment_basis.calculate(r_unit, rb_values)
        for coef, mlip_basis_value in zip(mtp_data.moment_coeffs, mlip_basis_values):
            bc_map[float(mlip_basis_value)] = coef

        # Calculate our basis for test vectors
        moment_basis = self.moment_basis
        basis = _calc_moment_basis(
            r_unit,
            rb_values,
            moment_basis.basic_moments,
            moment_basis.pair_contractions,
            moment_basis.scalar_contractions,
        )

        # Compare our basis with MLIP like basis
        remapped_coeffs = []
        basis_contractions_to_remove = []
        remaining_mlip_bs = list(bc_map.keys())

        relative_tolerance = 1e-8
        for basis_value, contraction in zip(basis, moment_basis.scalar_contractions):
            for mlip_basis_value, coef in bc_map.items():
                if np.isclose(mlip_basis_value, basis_value, rtol=relative_tolerance):
                    remapped_coeffs.append(coef)
                    remaining_mlip_bs.remove(mlip_basis_value)
                    break
            else:
                warn(
                    "Basis contraction was not found in the MLIP file. "
                    f"It will now be omitted from the basis.\n{contraction}: {basis_value}"
                )
                basis_contractions_to_remove.append(contraction)

        if len(remaining_mlip_bs) > 0:
            raise RuntimeError(
                "Not all MLIP contractions found:\n" f"{remaining_mlip_bs}\n"
            )

        # Remove contractions not present in the MLIP potential file
        # [should rarely be needed, as they now agree perfectly (tested at least to lvl 22)]
        for contraction in basis_contractions_to_remove:
            moment_basis.scalar_contractions.remove(contraction)

        self.remapped_coeffs = np.array(remapped_coeffs)


def _calc_moment_basis(
    r_unit,
    rb_values,
    basic_moments,
    pair_contractions,
    scalar_contractions,
):
    calculated_moments = _calc_basic_moments(r_unit, rb_values, basic_moments)
    for contraction in pair_contractions:
        m1 = calculated_moments[contraction[0]]
        m2 = calculated_moments[contraction[1]]
        # Contract two moments by tensordot
        calculated_contraction = np.tensordot(m1, m2, axes=contraction[3])
        calculated_moments[contraction] = calculated_contraction
    basis = []
    for inds in scalar_contractions:
        b = calculated_moments[inds]
        basis.append(b)
    return basis


def _calc_basic_moments(r_unit, rb_values, moment_descriptions):
    calculated_moments = {}
    for moment in moment_descriptions:
        mu, nu = moment[0:2]
        m = (rb_values[mu, :] * make_tensor(r_unit, nu).T).T.sum(axis=0)
        calculated_moments[moment] = m
    return calculated_moments
