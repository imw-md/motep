"""MTP writtin in Python.

Original version: Axel Forslund
Modified version: Yuji Ikeda
"""

import numpy as np
from ase import Atoms

from motep.potentials.mtp import get_types
from motep.potentials.mtp.base import EngineBase
from motep.potentials.mtp.data import MTPData

from .chebyshev import ChebyshevArrayRadialBasis
from .moment import MomentBasis


class NumpyMTPEngine(EngineBase):
    """MTP engine based on NumPy."""

    def __init__(self, mtp_data: MTPData, **kwargs: dict) -> None:
        """Intialize the engine."""
        self.rb = ChebyshevArrayRadialBasis(mtp_data)
        super().__init__(mtp_data, **kwargs)

    def update(self, mtp_data: MTPData) -> None:
        """Update MTP parameters."""
        super().update(mtp_data)
        if self.mtp_data.radial_coeffs is not None:
            self.rb.update_coeffs(self.mtp_data.radial_coeffs)

    def _calc_basis(
        self,
        i: int,
        itype: int,
        js: list[int],
        jtypes: list[int],
        r_ijs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        r_abs = np.sqrt(np.add.reduce(r_ijs**2, axis=1))
        r_ijs_unit = (r_ijs.T / r_abs).T

        self.rb.calc_radial_part(r_abs, itype, jtypes)
        np.add.at(self.rbd.values[itype], jtypes, self.rb.basis_vs.T)
        for k, (j, jtype) in enumerate(zip(js, jtypes, strict=True)):
            tmp = self.rb.basis_ds[:, k, None] * r_ijs_unit[k]
            self.rbd.dqdris[itype, jtype, :, i] -= tmp
            self.rbd.dqdris[itype, jtype, :, j] += tmp
            self.rbd.dqdeps[itype, jtype] += tmp[:, :, None] * r_ijs[k]
        moment_basis = MomentBasis(self.mtp_data)
        return moment_basis.calculate(itype, jtypes, r_ijs, r_abs, self.rb)

    def calculate(self, atoms: Atoms) -> tuple:
        """Calculate properties of the given system."""
        self.update_neighbor_list(atoms)
        itypes = get_types(atoms, self.mtp_data.species)
        energies = self.mtp_data.species_coeffs[itypes]

        self.mbd.clean()
        self.rbd.clean()

        moment_coeffs = self.mtp_data.moment_coeffs

        for i, itype in enumerate(itypes):
            js, r_ijs = self._get_distances(atoms, i)
            jtypes = itypes[js]
            basis_values, basis_jac_rs, dedcs, dgdcs = self._calc_basis(
                i,
                itype,
                js,
                jtypes,
                r_ijs,
            )

            self.mbd.values += basis_values

            energies[i] += moment_coeffs @ basis_values
            # Calculate forces
            # Be careful that the derivative of the site energy of the j-th atom also
            # contributes to the forces on the i-th atom.
            # Be also careful that:
            # 1. In `calc_moment_basis`, the derivatives with respect to the j-th atom
            #    (not the center i-th) atom is computed.
            # 2. The force on the i-th atom is defined as the negative of the gradient
            #    with respect to the i-th atom.
            # Thus, the negative signs of the two contributions are cancelled out below.
            for k, j in enumerate(js):
                self.mbd.dbdris[:, i] -= basis_jac_rs[:, k]
                self.mbd.dbdris[:, j] += basis_jac_rs[:, k]
            self.mbd.dbdeps += r_ijs.T @ basis_jac_rs

            self.mbd.dedcs[itype] += dedcs

            for k, j in enumerate(js):
                self.mbd.dgdcs[itype, :, :, :, i] -= dgdcs[:, :, :, k]
                self.mbd.dgdcs[itype, :, :, :, j] += dgdcs[:, :, :, k]
            self.mbd.dsdcs[itype] += r_ijs.T @ dgdcs

        forces = np.sum(moment_coeffs * self.mbd.dbdris.T, axis=-1).T * -1.0
        stress = np.sum(moment_coeffs * self.mbd.dbdeps.T, axis=-1).T

        self.results["energies"] = energies
        self.results["energy"] = self.results["energies"].sum()
        self.results["forces"] = forces

        self._symmetrize_stress(atoms, stress)

        self.results["stress"] = stress.flat[[0, 4, 8, 5, 2, 1]]

        return self.results["energy"], self.results["forces"], self.results["stress"]
