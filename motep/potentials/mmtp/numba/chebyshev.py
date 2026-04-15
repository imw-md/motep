"""Radial basis functions based on Chebyshev polynomials."""

import numba as nb
import numpy as np
import numpy.typing as npt

from motep.potentials.mtp.numba.chebyshev import chebyshev


@nb.njit(
    nb.types.Tuple(
        (
            nb.float64[:, :],
            nb.float64[:, :],
            nb.float64[:, :],
            nb.float64[:, :],
        ),
    )(
        nb.float64[:],
        nb.float64[:],
        nb.int32,
        nb.int32,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
    ),
    cache=True,
)
def calc_radial_and_mag_basis(
    r_abs: npt.NDArray[np.float64],
    ms: npt.NDArray[np.float64],
    radial_basis_size: int,
    magnetic_basis_size: int,
    scaling: float,
    min_dist: float,
    max_dist: float,
    mi: float,
    min_mag: float,
    max_mag: float,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Calculate combined radial+magnetic basis values and their derivatives."""
    rbs = radial_basis_size
    mbs = magnetic_basis_size
    nrb = rbs * mbs * mbs
    basis = np.zeros((nrb, r_abs.size))
    dbdrs = np.zeros((nrb, r_abs.size))
    dbdmis = np.zeros((nrb, r_abs.size))
    dbdmjs = np.zeros((nrb, r_abs.size))

    for j in range(r_abs.size):
        if r_abs[j] < max_dist:

            smooth_value = scaling * (max_dist - r_abs[j]) ** 2
            smooth_deriv = -2.0 * scaling * (max_dist - r_abs[j])
            br, dr = chebyshev(r_abs[j], rbs, min_dist, max_dist)

            bm1, dm1 = chebyshev(mi, mbs, min_mag, max_mag)
            bm2, dm2 = chebyshev(ms[j], mbs, min_mag, max_mag)
            for irb in range(rbs):
                for i1 in range(mbs):
                    for i2 in range(mbs):
                        val = br[irb] * bm1[i1] * bm2[i2]
                        der = dr[irb] * bm1[i1] * bm2[i2]
                        ind = irb * mbs * mbs + i1 * mbs + i2
                        basis[ind, j] = val * smooth_value
                        dbdrs[ind, j] = der * smooth_value + val * smooth_deriv
                        dbdmis[ind, j] = br[irb] * dm1[i1] * bm2[i2] * smooth_value
                        dbdmjs[ind, j] = br[irb] * bm1[i1] * dm2[i2] * smooth_value

    return basis, dbdrs, dbdmis, dbdmjs
