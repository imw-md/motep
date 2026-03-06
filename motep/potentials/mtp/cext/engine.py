"""MTP Engine using C extension."""

import numpy as np
import numpy.typing as npt
from ase import Atoms

from motep.potentials.mtp import get_types
from motep.potentials.mtp.base import EngineBase

try:
    from . import _mtp_cext
except ImportError as e:
    raise ImportError(
        "C extension module '_mtp_cext' not found. "
        "Please build the extension with: pip install -e ."
    ) from e


class CExtMTPEngine(EngineBase):
    """MTP Engine based on C extension.

    This engine provides similar functionality to the numba-based engine
    but uses compiled C code for better performance in some scenarios.
    """

    def __init__(self, *args: tuple, **kwargs: dict) -> None:
        """Initialize the engine."""
        super().__init__(*args, **kwargs)

    def _calculate(self, atoms: Atoms) -> tuple:
        """Main calculation dispatch."""
        if self.mode == "run":
            return self._calc_run(atoms)
        if self.mode == "train":
            return self._calc_train(atoms)
        raise NotImplementedError(self.mode)

    def _calc_run(self, atoms: Atoms) -> tuple:
        """Calculate energies, forces, and stress for run mode."""
        mtp_data = self.mtp_data

        all_js = self._neighbors
        all_r_ijs = self._get_interatomic_vectors(atoms)

        itypes = get_types(atoms, mtp_data.species)
        all_jtypes = itypes[all_js]

        self.mbd.clean()
        self.rbd.clean()

        energies, gradient = _mtp_cext.calc_run(
            all_js,
            all_r_ijs,
            itypes,
            all_jtypes,
            mtp_data,
            self.mbd,  # output
        )

        forces = _mtp_cext.calc_forces_from_gradient(gradient, all_js)
        stress = np.einsum("ijk, ijl -> lk", all_r_ijs, gradient)

        return energies, forces, stress

    def _calc_train(self, atoms: Atoms) -> tuple:
        """Calculate energies, forces, and stress for training mode."""
        mtp_data = self.mtp_data

        js = self._neighbors
        rs = self._get_interatomic_vectors(atoms)

        itypes = get_types(atoms, mtp_data.species)
        jtypes = itypes[js]

        self.rbd.clean()
        self.mbd.clean()

        energies = _mtp_cext.calc_train(
            js,
            rs,
            itypes,
            jtypes,
            mtp_data,
            self.rbd,  # output
            self.mbd,  # output
        )

        moment_coeffs = mtp_data.moment_coeffs

        forces = np.sum(moment_coeffs * self.mbd.dbdris.T, axis=-1).T * -1.0
        stress = np.sum(moment_coeffs * self.mbd.dbdeps.T, axis=-1).T

        return energies, forces, stress
