"""MTP written using numba."""

import numba as nb
import numpy as np
import numpy.typing as npt
from ase import Atoms

from motep.potentials.mtp import get_types
from motep.potentials.mtp.base import EngineBase

from .chebyshev import _nb_calc_radial_basis, _nb_calc_radial_funcs
from .utils import (
    _nb_calc_local_energy_and_gradient,
    _nb_calc_moment,
    _nb_forces_from_gradient,
    _store_radial_basis_values,
)


class NumbaMTPEngine(EngineBase):
    """MTP Engine based on Numba."""

    def __init__(self, *args: tuple, **kwargs: dict) -> None:
        """Intialize the engine."""
        super().__init__(*args, **kwargs)
        self.calculate = self._calc_train if self._is_trained else self._calc_run

    def _calc_run(self, atoms: Atoms) -> tuple:
        """Calculate properties of the given system."""
        self.update_neighbor_list(atoms)

        mtp_data = self.mtp_data

        all_js, all_r_ijs = self._get_all_distances(atoms)
        itypes = get_types(atoms, self.mtp_data.species)
        all_jtypes = itypes[all_js]
        energies, gradient = _nb_calc_energy_and_gradient(
            all_r_ijs,
            itypes,
            all_jtypes,
            mtp_data.alpha_moments_count,
            mtp_data.alpha_moment_mapping,
            mtp_data.alpha_index_basic,
            mtp_data.alpha_index_times,
            mtp_data.scaling,
            mtp_data.min_dist,
            mtp_data.max_dist,
            mtp_data.radial_coeffs,
            mtp_data.species_coeffs,
            mtp_data.moment_coeffs,
        )

        forces = _nb_forces_from_gradient(gradient, all_js, all_js.shape[1])
        stress = np.einsum("ijk, ijl -> lk", all_r_ijs, gradient)

        self._symmetrize_stress(atoms, stress)

        self.results["energies"] = energies
        self.results["energy"] = self.results["energies"].sum()
        self.results["forces"] = forces
        self.results["stress"] = stress.flat[[0, 4, 8, 5, 2, 1]]

        return self.results

    def _calc_train(self, atoms: Atoms) -> tuple:
        self.update_neighbor_list(atoms)

        mtp_data = self.mtp_data

        itypes = get_types(atoms, self.mtp_data.species)
        energies = self.mtp_data.species_coeffs[itypes]

        self.mbd.clean()
        self.rbd.clean()

        moment_coeffs = mtp_data.moment_coeffs

        stress = np.zeros((3, 3))
        for i, itype in enumerate(itypes):
            js, r_ijs = self._get_distances(atoms, i)
            jtypes = itypes[js]
            r_abs = _nb_linalg_norm(r_ijs)
            r_ijs_unit = _calc_r_unit(r_ijs, r_abs)
            rb_values, rb_derivs = _nb_calc_radial_basis(
                r_abs,
                mtp_data.radial_basis_size,
                mtp_data.scaling,
                mtp_data.min_dist,
                mtp_data.max_dist,
            )
            _store_radial_basis_values(
                i,
                itype,
                js,
                jtypes,
                r_ijs,
                r_ijs_unit,
                rb_values,
                rb_derivs,
                self.rbd.values,
                self.rbd.dqdris,
                self.rbd.dqdeps,
            )
            basis_values, basis_jac_rs, dedcs, dgdcs = _nb_calc_moment(
                itype,
                jtypes,
                r_abs,
                r_ijs_unit,
                rb_values,
                rb_derivs,
                mtp_data.radial_coeffs,
                mtp_data.alpha_moments_count,
                mtp_data.alpha_moment_mapping,
                mtp_data.alpha_index_basic,
                mtp_data.alpha_index_times,
                mtp_data.moment_coeffs,
            )

            energies[i] += moment_coeffs @ basis_values

            _update_mbd_values(self.mbd.values, basis_values)
            _update_mbd_dbdris(i, js, self.mbd.dbdris, basis_jac_rs)
            _update_mbd_dbdeps(js, r_ijs, self.mbd.dbdeps, basis_jac_rs)
            _update_mbd_dedcs(itype, self.mbd.dedcs, dedcs)
            _update_mbd_dgdcs(i, itype, js, self.mbd.dgdcs, dgdcs)
            _update_mbd_dsdcs(itype, js, r_ijs, self.mbd.dsdcs, dgdcs)

        forces = np.sum(moment_coeffs * self.mbd.dbdris.T, axis=-1).T * -1.0
        stress = np.sum(moment_coeffs * self.mbd.dbdeps.T, axis=-1).T

        self._symmetrize_stress(atoms, stress)

        stress = stress.flat[[0, 4, 8, 5, 2, 1]]

        self.results["energies"] = energies
        self.results["energy"] = self.results["energies"].sum()
        self.results["forces"] = forces
        self.results["stress"] = stress

        return self.results


@nb.njit(nb.float64[:](nb.float64[:, :]))
def _nb_linalg_norm(r_ijs: np.ndarray) -> np.ndarray:
    r_abs = np.zeros((r_ijs.shape[0],))
    for j in range(r_ijs.shape[0]):
        for k in range(3):
            r_abs[j] += r_ijs[j, k] ** 2
        r_abs[j] **= 0.5
    return r_abs


@nb.njit(nb.float64[:, :](nb.float64[:, :], nb.float64[:]))
def _calc_r_unit(r_ijs: np.ndarray, r_abs: np.ndarray) -> np.ndarray:
    r_ijs_unit = np.zeros((r_ijs.shape[0], 3))
    for j in range(r_ijs.shape[0]):
        for k in range(3):
            r_ijs_unit[j, k] = r_ijs[j, k] / r_abs[j]
    return r_ijs_unit


@nb.njit((nb.float64[:], nb.float64[:]))
def _update_mbd_values(
    mbd_values: npt.NDArray[np.float64],
    basis_values: npt.NDArray[np.float64],
) -> None:
    for iamc in range(mbd_values.size):
        mbd_values[iamc] += basis_values[iamc]


@nb.njit((nb.int64, nb.int64[:], nb.float64[:, :, :], nb.float64[:, :, :]))
def _update_mbd_dbdris(
    i: np.int64,
    js: npt.NDArray[np.int64],
    mbd_dbdris: npt.NDArray[np.float64],
    basis_jac_rs: npt.NDArray[np.float64],
) -> None:
    for iamc in range(mbd_dbdris.shape[0]):
        for k, j in enumerate(js):
            for ixyz0 in range(3):
                mbd_dbdris[iamc, i, ixyz0] -= basis_jac_rs[iamc, k, ixyz0]
                mbd_dbdris[iamc, j, ixyz0] += basis_jac_rs[iamc, k, ixyz0]


@nb.njit((nb.int64[:], nb.float64[:, :], nb.float64[:, :, :], nb.float64[:, :, :]))
def _update_mbd_dbdeps(
    js: npt.NDArray[np.int64],
    r_ijs: npt.NDArray[np.float64],
    mbd_dbdeps: npt.NDArray[np.float64],
    basis_jac_rs: npt.NDArray[np.float64],
) -> None:
    for iamc in range(mbd_dbdeps.shape[0]):
        for k in range(js.size):
            for ixyz0 in range(3):
                for ixyz1 in range(3):
                    mbd_dbdeps[iamc, ixyz0, ixyz1] += (
                        r_ijs[k, ixyz0] * basis_jac_rs[iamc, k, ixyz1]
                    )


@nb.njit((nb.int64, nb.float64[:, :, :, :], nb.float64[:, :, :]))
def _update_mbd_dedcs(
    itype: np.int64,
    mbd_dedcs: npt.NDArray[np.float64],
    tmp_dedcs: npt.NDArray[np.float64],
) -> None:
    _, s1, s2, s3 = mbd_dedcs.shape
    for i1 in range(s1):
        for i2 in range(s2):
            for i3 in range(s3):
                mbd_dedcs[itype, i1, i2, i3] += tmp_dedcs[i1, i2, i3]


@nb.njit(
    (
        nb.int64,
        nb.int64,
        nb.int64[:],
        nb.float64[:, :, :, :, :, :],
        nb.float64[:, :, :, :, :],
    ),
)
def _update_mbd_dgdcs(
    i: np.int64,
    itype: np.int64,
    js: npt.NDArray[np.int64],
    mbd_dgdcs: npt.NDArray[np.float64],
    tmp_dgdcs: npt.NDArray[np.float64],
) -> None:
    s1, s2, s3 = mbd_dgdcs.shape[1:4]
    for i1 in range(s1):
        for i2 in range(s2):
            for i3 in range(s3):
                for k, j in enumerate(js):
                    for ixyz0 in range(3):
                        v = tmp_dgdcs[i1, i2, i3, k, ixyz0]
                        mbd_dgdcs[itype, i1, i2, i3, i, ixyz0] -= v
                        mbd_dgdcs[itype, i1, i2, i3, j, ixyz0] += v


@nb.njit(
    (
        nb.int64,
        nb.int64[:],
        nb.float64[:, :],
        nb.float64[:, :, :, :, :, :],
        nb.float64[:, :, :, :, :],
    ),
)
def _update_mbd_dsdcs(
    itype: np.int64,
    js: npt.NDArray[np.int64],
    r_ijs: npt.NDArray[np.float64],
    mbd_dsdcs: npt.NDArray[np.float64],
    tmp_dgdcs: npt.NDArray[np.float64],
) -> None:
    s1, s2, s3 = mbd_dsdcs.shape[1:4]
    for i1 in range(s1):
        for i2 in range(s2):
            for i3 in range(s3):
                for k in range(js.size):
                    for ixyz0 in range(3):
                        for ixyz1 in range(3):
                            v = r_ijs[k, ixyz0] * tmp_dgdcs[i1, i2, i3, k, ixyz1]
                            mbd_dsdcs[itype, i1, i2, i3, ixyz0, ixyz1] += v


@nb.njit(
    nb.types.Tuple((nb.float64[:], nb.float64[:, :, :]))(
        nb.float64[:, :, :],
        nb.int64[:],
        nb.int64[:, :],
        nb.int64,
        nb.int64[:],
        nb.int64[:, :],
        nb.int64[:, :],
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64[:, :, :, :],
        nb.float64[:],
        nb.float64[:],
    ),
    # parallel=True,
)
def _nb_calc_energy_and_gradient(
    all_r_ijs: npt.NDArray[np.float64],
    itypes: npt.NDArray[np.int64],
    all_jtypes: npt.NDArray[np.int64],
    alpha_moments_count: np.int64,
    alpha_moment_mapping: npt.NDArray[np.int64],
    alpha_index_basic: npt.NDArray[np.int64],
    alpha_index_times: npt.NDArray[np.int64],
    scaling: np.float64,
    min_dist: np.float64,
    max_dist: np.float64,
    radial_coeffs: npt.NDArray[np.float64],
    species_coeffs: npt.NDArray[np.float64],
    moment_coeffs: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    energies = species_coeffs[itypes]
    gradient = np.zeros((itypes.size, all_jtypes.shape[1], 3))
    for i in nb.prange(itypes.size):
        r_abs = _nb_linalg_norm(all_r_ijs[i, :, :])
        r_ijs_unit = _calc_r_unit(all_r_ijs[i, :, :], r_abs)
        rb_values, rb_derivs = _nb_calc_radial_funcs(
            r_abs,
            itypes[i],
            all_jtypes[i, :],
            radial_coeffs,
            scaling,
            min_dist,
            max_dist,
        )
        local_energy, local_gradient = _nb_calc_local_energy_and_gradient(
            r_ijs_unit,
            r_abs,
            rb_values,
            rb_derivs,
            alpha_moments_count,
            alpha_moment_mapping,
            alpha_index_basic,
            alpha_index_times,
            moment_coeffs,
        )
        energies[i] += local_energy
        for j in range(all_jtypes.shape[1]):
            for k in range(3):
                gradient[i, j, k] = local_gradient[j, k]
    return energies, gradient
