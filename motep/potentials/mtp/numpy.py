"""MTP writtin in Python.

Original version: Axel Forslund
Modified version: Yuji Ikeda
"""

import numpy as np
import numpy.typing as npt
from ase import Atoms
from ase.neighborlist import PrimitiveNeighborList

from motep.potentials.mtp.base import EngineBase
from motep.potentials.mtp.data import MTPData
from motep.potentials.mtp.moment import MomentBasis
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


def get_types(atoms: Atoms, species: list[int]) -> npt.NDArray[np.int64]:
    return np.fromiter((species.index(_) for _ in atoms.numbers), dtype=int)


class NumpyMTPEngine(EngineBase):
    """MTP engine based on NumPy."""

    def __init__(self, dict_mtp: MTPData | None = None) -> None:
        """Intialize the engine.

        Parameters
        ----------
        dict_mtp : :class:`motep.potentials.mtp.data.MTPData`
            Parameters in the MLIP .mtp file.

        """
        self.rb = ChebyshevArrayRadialBasis(dict_mtp)
        super().__init__(dict_mtp)

    def update(self, dict_mtp: MTPData) -> None:
        """Update MTP parameters."""
        super().update(dict_mtp)
        if "radial_coeffs" in self.dict_mtp:
            self.rb.update_coeffs(self.dict_mtp["radial_coeffs"])

    def _calc_basis(
        self,
        i: int,
        itype: int,
        js: list[int],
        jtypes: list[int],
        r_ijs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        r_abs = np.sqrt(np.add.reduce(r_ijs**2, axis=0))
        r_ijs_unit = r_ijs / r_abs

        self.rb.calc_radial_part(r_abs, itype, jtypes)
        np.add.at(self.radial_basis_values[itype], jtypes, self.rb.basis_vs[:, :])
        for k, (j, jtype) in enumerate(zip(js, jtypes, strict=True)):
            tmp = self.rb.basis_ds[k, :, None] * r_ijs_unit[:, k]
            self.radial_basis_dqdris[itype, jtype, :, :, i] -= tmp
            self.radial_basis_dqdris[itype, jtype, :, :, j] += tmp
            self.radial_basis_dqdeps[itype, jtype] += tmp[:, :, None] * r_ijs[:, k]
        moment_basis = MomentBasis(self.dict_mtp)
        return moment_basis.calculate(itype, jtypes, r_ijs, r_abs, self.rb)

    def calculate(self, atoms: Atoms) -> tuple:
        """Calculate properties of the given system."""
        self.update_neighbor_list(atoms)
        itypes = get_types(atoms, self.dict_mtp["species"])
        self.energies = self.dict_mtp["species_coeffs"][itypes]

        self.basis_values[:] = 0.0
        self.basis_dbdris[:, :, :] = 0.0
        self.basis_dbdeps[:, :, :] = 0.0
        self.basis_de_dcs[...] = 0.0
        self.basis_ddedcs[...] = 0.0
        self.basis_ds_dcs[...] = 0.0

        self.radial_basis_values[...] = 0.0
        self.radial_basis_dqdris[...] = 0.0
        self.radial_basis_dqdeps[...] = 0.0

        moment_coeffs = self.dict_mtp["moment_coeffs"]

        for i, itype in enumerate(itypes):
            js, r_ijs = self._get_distances(atoms, i)
            jtypes = [self.dict_mtp["species"].index(atoms.numbers[j]) for j in js]
            basis_values, basis_jac_rs, basis_jac_cs, basis_jac_rc = self._calc_basis(
                i,
                itype,
                js,
                jtypes,
                r_ijs,
            )

            self.basis_values += basis_values

            self.energies[i] += moment_coeffs @ basis_values
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
                self.basis_dbdris[:, :, i] -= basis_jac_rs[:, :, k]
                self.basis_dbdris[:, :, j] += basis_jac_rs[:, :, k]
            self.basis_dbdeps += basis_jac_rs @ r_ijs.T

            self.basis_de_dcs[itype] += (moment_coeffs * basis_jac_cs.T).sum(axis=-1).T

            tmp = (moment_coeffs * basis_jac_rc.T).sum(axis=-1).T
            for k, j in enumerate(js):
                self.basis_ddedcs[itype, :, :, :, :, i] -= tmp[:, :, :, :, k]
                self.basis_ddedcs[itype, :, :, :, :, j] += tmp[:, :, :, :, k]
            self.basis_ds_dcs[itype] += tmp @ r_ijs.T

        self.forces = np.sum(moment_coeffs * self.basis_dbdris.T, axis=-1) * -1.0
        self.stress = np.sum(moment_coeffs * self.basis_dbdeps.T, axis=-1).T

        self.results["energies"] = self.energies
        self.results["energy"] = self.results["energies"].sum()
        self.results["forces"] = self.forces

        if atoms.cell.rank == 3:
            volume = atoms.get_volume()
            self.stress = (self.stress + self.stress.T) * 0.5  # symmetrize
            self.stress /= volume
            self.basis_dbdeps += self.basis_dbdeps.transpose(0, 2, 1)
            self.basis_dbdeps *= 0.5 / volume
            self.basis_ds_dcs += self.basis_ds_dcs.swapaxes(-2, -1)
            self.basis_ds_dcs *= 0.5 / volume
            axes = 0, 1, 2, 4, 3
            self.radial_basis_dqdeps += self.radial_basis_dqdeps.transpose(axes)
            self.radial_basis_dqdeps *= 0.5 / volume
        else:
            self.stress[:, :] = np.nan
            self.basis_dbdeps[:, :, :] = np.nan
            self.radial_basis_dqdeps[:, :, :] = np.nan

        self.results["stress"] = self.stress.flat[[0, 4, 8, 5, 2, 1]]

        return self.results["energy"], self.results["forces"], self.results["stress"]

    def jac_energy(self, atoms: Atoms) -> MTPData:
        """Calculate the Jacobian of the energy with respect to the MTP parameters."""
        sps = self.dict_mtp["species"]
        nbs = list(atoms.numbers)

        jac = MTPData()  # placeholder of the Jacobian with respect to the parameters
        jac["scaling"] = 0.0
        jac["moment_coeffs"] = self.basis_values.copy()
        jac["species_coeffs"] = np.fromiter((nbs.count(s) for s in sps), dtype=float)
        jac["radial_coeffs"] = self.basis_de_dcs.copy()

        return jac

    def jac_forces(self, atoms: Atoms) -> MTPData:
        """Calculate the Jacobian of the forces with respect to the MTP parameters.

        `jac.parameters` have the shape of `(nparams, natoms, 3)`.

        """
        spc = self.dict_mtp["species_count"]
        number_of_atoms = len(atoms)

        jac = Jac()  # placeholder of the Jacobian with respect to the parameters
        jac["scaling"] = np.zeros((1, number_of_atoms, 3))
        jac["moment_coeffs"] = self.basis_dbdris.transpose(0, 2, 1) * -1.0
        jac["species_coeffs"] = np.zeros((spc, number_of_atoms, 3))
        jac["radial_coeffs"] = self.basis_ddedcs.transpose(0, 1, 2, 3, 5, 4) * -1.0

        return jac

    def jac_stress(self, atoms: Atoms) -> MTPData:
        """Calculate the Jacobian of the forces with respect to the MTP parameters.

        `jac.parameters` have the shape of `(nparams, natoms, 3)`.

        """
        spc = self.dict_mtp["species_count"]

        jac = Jac()  # placeholder of the Jacobian with respect to the parameters
        jac["scaling"] = np.zeros((1, 3, 3))
        jac["moment_coeffs"] = self.basis_dbdeps.copy()
        jac["species_coeffs"] = np.zeros((spc, 3, 3))
        jac["radial_coeffs"] = self.basis_ds_dcs.copy()

        return jac


#
# Class for Numba implementation
#
class NumbaMTPEngine(EngineBase):
    """MTP Engine based on Numba."""

    def calculate(self, atoms: Atoms):
        self.update_neighbor_list(atoms)
        return self.numba_calc_energy_and_forces(atoms)

    def numba_calc_energy_and_forces(self, atoms):
        from .numba import numba_calc_energy_and_forces

        mlip_params = self.dict_mtp
        energy, forces, stress = numba_calc_energy_and_forces(
            self,
            atoms,
            mlip_params["alpha_moments_count"],
            mlip_params["alpha_moment_mapping"],
            mlip_params["alpha_index_basic"],
            mlip_params["alpha_index_times"],
            mlip_params["scaling"],
            mlip_params["min_dist"],
            mlip_params["max_dist"],
            mlip_params["species_coeffs"],
            mlip_params["moment_coeffs"],
            mlip_params["radial_coeffs"],
        )
        return energy, forces, stress
