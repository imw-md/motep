"""MTP writtin in Python.

Original version: Axel Forslund
Modified version: Yuji Ikeda
"""

import numpy as np
import numpy.typing as npt
from ase import Atoms

from motep.potentials.mtp import get_types
from motep.potentials.mtp.base import EngineBase
from motep.potentials.mtp.data import MTPData
from motep.potentials.mtp.numpy.moment import MomentBasis
from motep.radial import ChebyshevArrayRadialBasis


class Jac(dict):
    @property
    def parameters(self) -> npt.NDArray[np.float64]:
        shape = self["radial_coeffs"].shape
        return np.concatenate(
            (
                self["scaling"],
                self["moment_coeffs"],
                self["species_coeffs"],
                self["radial_coeffs"].reshape(-1, *shape[4::]),
            ),
        )


class NumpyMTPEngine(EngineBase):
    """MTP engine based on NumPy."""

    def __init__(self, mtp_data: MTPData, **kwargs: dict) -> None:
        """Intialize the engine."""
        self.rb = ChebyshevArrayRadialBasis(mtp_data)
        super().__init__(mtp_data, **kwargs)

    def update(self, mtp_data: MTPData) -> None:
        """Update MTP parameters."""
        super().update(mtp_data)
        if "radial_coeffs" in self.mtp_data:
            self.rb.update_coeffs(self.mtp_data["radial_coeffs"])

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
        np.add.at(self.rbd.values[itype], jtypes, self.rb.basis_vs[:, :])
        for k, (j, jtype) in enumerate(zip(js, jtypes, strict=True)):
            tmp = self.rb.basis_ds[k, :, None] * r_ijs_unit[k]
            self.rbd.dqdris[itype, jtype, :, :, i] -= tmp
            self.rbd.dqdris[itype, jtype, :, :, j] += tmp
            self.rbd.dqdeps[itype, jtype] += tmp[:, :, None] * r_ijs[k]
        moment_basis = MomentBasis(self.mtp_data)
        return moment_basis.calculate(itype, jtypes, r_ijs, r_abs, self.rb)

    def calculate(self, atoms: Atoms) -> tuple:
        """Calculate properties of the given system."""
        self.update_neighbor_list(atoms)
        itypes = get_types(atoms, self.mtp_data["species"])
        energies = self.mtp_data["species_coeffs"][itypes]

        self.mbd.clean()
        self.rbd.clean()

        moment_coeffs = self.mtp_data["moment_coeffs"]

        for i, itype in enumerate(itypes):
            js, r_ijs = self._get_distances(atoms, i)
            jtypes = [self.mtp_data["species"].index(atoms.numbers[j]) for j in js]
            basis_values, basis_jac_rs, basis_jac_cs, basis_jac_rc = self._calc_basis(
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

            self.mbd.de_dcs[itype] += (moment_coeffs * basis_jac_cs.T).sum(axis=-1).T

            tmp = (moment_coeffs * basis_jac_rc.T).sum(axis=-1).T
            for k, j in enumerate(js):
                self.mbd.ddedcs[itype, :, :, :, i] -= tmp[:, :, :, k]
                self.mbd.ddedcs[itype, :, :, :, j] += tmp[:, :, :, k]
            self.mbd.ds_dcs[itype] += r_ijs.T @ tmp

        forces = np.sum(moment_coeffs * self.mbd.dbdris.T, axis=-1).T * -1.0
        stress = np.sum(moment_coeffs * self.mbd.dbdeps.T, axis=-1).T

        self.results["energies"] = energies
        self.results["energy"] = self.results["energies"].sum()
        self.results["forces"] = forces

        self._symmetrize_stress(atoms, stress)

        self.results["stress"] = stress.flat[[0, 4, 8, 5, 2, 1]]

        return self.results["energy"], self.results["forces"], self.results["stress"]

    def jac_energy(self, atoms: Atoms) -> MTPData:
        """Calculate the Jacobian of the energy with respect to the MTP parameters."""
        sps = self.mtp_data["species"]
        nbs = list(atoms.numbers)

        jac = MTPData()  # placeholder of the Jacobian with respect to the parameters
        jac["scaling"] = 0.0
        jac["moment_coeffs"] = self.mbd.values.copy()
        jac["species_coeffs"] = np.fromiter((nbs.count(s) for s in sps), dtype=float)
        jac["radial_coeffs"] = self.mbd.de_dcs.copy()

        return jac

    def jac_forces(self, atoms: Atoms) -> MTPData:
        """Calculate the Jacobian of the forces with respect to the MTP parameters.

        `jac.parameters` have the shape of `(nparams, natoms, 3)`.

        """
        spc = self.mtp_data["species_count"]
        number_of_atoms = len(atoms)

        jac = Jac()  # placeholder of the Jacobian with respect to the parameters
        jac["scaling"] = np.zeros((1, number_of_atoms, 3))
        jac["moment_coeffs"] = self.mbd.dbdris * -1.0
        jac["species_coeffs"] = np.zeros((spc, number_of_atoms, 3))
        jac["radial_coeffs"] = self.mbd.ddedcs * -1.0

        return jac

    def jac_stress(self, atoms: Atoms) -> MTPData:
        """Calculate the Jacobian of the forces with respect to the MTP parameters.

        `jac.parameters` have the shape of `(nparams, natoms, 3)`.

        """
        spc = self.mtp_data["species_count"]

        jac = Jac()  # placeholder of the Jacobian with respect to the parameters
        jac["scaling"] = np.zeros((1, 3, 3))
        jac["moment_coeffs"] = self.mbd.dbdeps.copy()
        jac["species_coeffs"] = np.zeros((spc, 3, 3))
        jac["radial_coeffs"] = self.mbd.ds_dcs.copy()

        return jac
