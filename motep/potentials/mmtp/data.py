"""Initializer."""

import logging
from dataclasses import dataclass, field, fields, replace
from typing import Self

import numpy as np

from motep.potentials.mtp.data import BasisData, MTPData

logger = logging.getLogger(__name__)


@dataclass
class MagMTPData(MTPData):
    """Subclass of `dict` to handle MTP parameters."""

    radial_basis: BasisData = field(
        default_factory=lambda: BasisData(
            type="RBChebyshev",
            min=np.float64(1.9),
            max=np.float64(5.5),
            size=np.int32(8),
        )
    )
    magnetic_basis: BasisData = field(
        default_factory=lambda: BasisData(
            type="BChebyshev",
            min=np.float64(-3.5),
            max=np.float64(3.5),
            size=np.int32(2),
        )
    )

    @classmethod
    def from_base(cls, obj: MTPData) -> Self:
        """Convert base class instance to MagMTPData.

        Returns:
            MagMTPData instance with fields from the base MTPData object.

        """
        return cls(**{_.name: getattr(obj, _.name) for _ in fields(obj)})

    def __post_init__(self) -> None:
        """Convert dict basis configs to BasisData objects if needed."""
        super().__post_init__()
        if isinstance(self.magnetic_basis, dict):
            default = self.__dataclass_fields__["magnetic_basis"].default_factory()
            self.magnetic_basis = replace(default, **self.magnetic_basis)

    # Backward-compatible properties
    @property
    def mag_basis_size(self) -> int:
        """Get magnetic basis size."""
        return self.magnetic_basis.size

    @mag_basis_size.setter
    def mag_basis_size(self, value: int) -> None:
        """Set magnetic basis size."""
        self.magnetic_basis.size = np.int32(value)

    @property
    def min_mag(self) -> float:
        """Get minimum magnetic value."""
        return float(self.magnetic_basis.min)

    @min_mag.setter
    def min_mag(self, value: float | np.float64) -> None:
        """Set minimum magnetic value."""
        self.magnetic_basis.min = np.float64(value)

    @property
    def max_mag(self) -> float:
        """Get maximum magnetic value."""
        return float(self.magnetic_basis.max)

    @max_mag.setter
    def max_mag(self, value: float | np.float64) -> None:
        """Set maximum magnetic value."""
        self.magnetic_basis.max = np.float64(value)

    def initialize(self, rng: np.random.Generator) -> None:
        """Initialize MTP parameters.

        Ensures ``radial_basis_size`` is the combined value ``rbs * mbs²`` and
        creates ``radial_coeffs`` with the correct combined shape.
        """
        if self.species_coeffs is None:
            self.species_coeffs = rng.uniform(-5.0, +5.0, self.species_count)
        if self.moment_coeffs is None:
            self.moment_coeffs = rng.uniform(-5.0, +5.0, self.alpha_scalar_moments)
        if self.radial_coeffs is None:
            spc = self.species_count
            rfc = self.radial_funcs_count
            mbs = self.mag_basis_size
            rbs_combined = self.radial_basis_size * mbs * mbs
            self.radial_coeffs = rng.uniform(-0.1, +0.1, (spc, spc, rfc, rbs_combined))

    @property
    def parameters(self) -> np.ndarray:
        """Serialized parameters."""
        tmp = []
        if "scaling" in self.optimized:
            tmp.append(np.atleast_1d(self.scaling))
        if "moment_coeffs" in self.optimized:
            tmp.append(self.moment_coeffs)
        if "species_coeffs" in self.optimized:
            tmp.append(self.species_coeffs)
        if "radial_coeffs" in self.optimized:
            tmp.append(self.radial_coeffs.flat)
        return np.hstack(tmp)

    @parameters.setter
    def parameters(self, parameters: list[float]) -> None:
        """Update data in the .mtp file.

        Parameters
        ----------
        parameters : list[float]
            MTP parameters.

        """
        species_count = self.species_count
        rfc = self.radial_funcs_count
        nrb = self.radial_basis_size * self.mag_basis_size**2
        asm = self.alpha_scalar_moments

        n = 0
        if "scaling" in self.optimized:
            self.scaling = parameters[n]
            n += 1
        if "moment_coeffs" in self.optimized:
            self.moment_coeffs = parameters[n : asm + n]
            n += asm
        if "species_coeffs" in self.optimized:
            self.species_coeffs = parameters[n : n + species_count]
            n += species_count
        if "radial_coeffs" in self.optimized:
            total_radial = parameters[n:]
            shape = species_count, species_count, rfc, nrb
            self.radial_coeffs = np.array(total_radial).reshape(shape)

    @property
    def number_of_parameters_optimized(self) -> int:
        """Get number of parameters optimized."""
        species_count = self.species_count
        rfc = self.radial_funcs_count
        nrb = self.radial_basis_size * self.mag_basis_size**2
        asm = self.alpha_scalar_moments
        n = 0
        if "scaling" in self.optimized:
            n += 1
        if "moment_coeffs" in self.optimized:
            n += asm
        if "species_coeffs" in self.optimized:
            n += species_count
        if "radial_coeffs" in self.optimized:
            n += species_count * species_count * rfc * nrb
        return n
