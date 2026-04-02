"""Magnetic MTP Engine using C extension."""

import types

import numpy as np
from ase import Atoms

from motep.potentials.mmtp.base import MagEngineBase
from motep.potentials.mtp.data import get_types

try:
    from . import _mmtp_cext
except ImportError as e:
    msg = (
        "C extension module '_mmtp_cext' not found. "
        "Please build the extension with: pip install -e ."
    )
    raise ImportError(msg) from e


class CExtMagMTPEngine(MagEngineBase):
    """Magnetic MTP Engine based on C extension.

    This engine provides similar functionality to the numba-based engine
    but uses compiled C code for better performance in some scenarios.
    """

    def __init__(self, *args: tuple, **kwargs: dict) -> None:
        """Initialize the engine."""
        super().__init__(*args, **kwargs)

    def _calculate(self, atoms: Atoms) -> tuple:
        """Main calculation dispatch."""
        if self.mode == "run":
            return self._calc_mag_run(atoms)
        if self.mode == "train":
            return self._calc_mag_train(atoms)
        if self.mode == "train_mgrad":
            return self._calc_mag_train_mgrad(atoms)
        raise NotImplementedError(self.mode)

    def _calc_mag_run(self, atoms: Atoms) -> tuple:
        """Calculate energies, forces, stress, and magnetic gradients for run mode."""
        mtp_data = self.mtp_data

        js = self._neighbors
        rs = self._get_interatomic_vectors(atoms)
        magnetic_moments = atoms.get_initial_magnetic_moments()

        itypes = get_types(atoms, mtp_data.species)
        jtypes = itypes[js]

        self.mbd.clean()
        self.rbd.clean()

        energies, gradient, grad_mag_i, grad_mag_j = _mmtp_cext.calc_mag_run(
            js,
            rs,
            magnetic_moments,
            itypes,
            jtypes,
            mtp_data,
            self.mbd,
        )

        forces = _mmtp_cext.calc_forces_from_gradient(gradient, js)
        mgrad = _mmtp_cext.calc_mgrad_from_gradient(grad_mag_i, grad_mag_j, js)
        stress = np.einsum("ijk, ijl -> lk", rs, gradient)

        return energies, forces, stress, mgrad

    def _calc_mag_train(self, atoms: Atoms) -> tuple:
        """Calculate energies, forces, stress, and magnetic gradients for training mode."""
        mtp_data = self.mtp_data

        js = self._neighbors
        rs = self._get_interatomic_vectors(atoms)
        magnetic_moments = atoms.get_initial_magnetic_moments()

        itypes = get_types(atoms, mtp_data.species)
        jtypes = itypes[js]

        self.mbd.clean()
        self.rbd.clean()

        energies = _mmtp_cext.calc_mag_train(
            js,
            rs,
            magnetic_moments,
            itypes,
            jtypes,
            mtp_data,
            self.rbd,
            self.mbd,
        )

        # Use the run implementation to get the magnetic gradients,
        # which is cheap compared to the train call.
        # Use a throwaway mbd to avoid double-accumulating mbd.vatoms.
        _tmp_mbd = types.SimpleNamespace(vatoms=np.zeros_like(self.mbd.vatoms))
        _, _, grad_mag_i, grad_mag_j = _mmtp_cext.calc_mag_run(
            js,
            rs,
            magnetic_moments,
            itypes,
            jtypes,
            mtp_data,
            _tmp_mbd,
        )

        moment_coeffs = mtp_data.moment_coeffs

        forces = np.sum(moment_coeffs * self.mbd.dbdris.T, axis=-1).T * -1.0
        stress = np.sum(moment_coeffs * self.mbd.dbdeps.T, axis=-1).T
        mgrad = _mmtp_cext.calc_mgrad_from_gradient(grad_mag_i, grad_mag_j, js)

        return energies, forces, stress, mgrad

    def _calc_mag_train_mgrad(self, atoms: Atoms) -> tuple:
        """Calculate energies, forces, stress, and magnetic gradients for train_mgrad mode."""
        mtp_data = self.mtp_data

        js = self._neighbors
        rs = self._get_interatomic_vectors(atoms)
        magnetic_moments = atoms.get_initial_magnetic_moments()

        itypes = get_types(atoms, mtp_data.species)
        jtypes = itypes[js]

        self.rbd.clean()
        self.mbd.clean()

        energies = _mmtp_cext.calc_mag_train_mgrad(
            js,
            rs,
            magnetic_moments,
            itypes,
            jtypes,
            mtp_data,
            self.rbd,
            self.mbd,
        )

        moment_coeffs = mtp_data.moment_coeffs

        forces = np.sum(moment_coeffs * self.mbd.dbdris.T, axis=-1).T * -1.0
        stress = np.sum(moment_coeffs * self.mbd.dbdeps.T, axis=-1).T
        mgrad = np.sum(moment_coeffs * self.mbd.dbdmis.T, axis=-1).T

        return energies, forces, stress, mgrad
