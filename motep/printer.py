"""Module for printing MTP parameters."""

from typing import Any

import numpy as np


class Printer:
    """Temporary class to print the MTP parameters."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    def print(self, parameters: np.ndarray) -> None:
        """Print parameters."""
        species_count = self.data["species_count"]
        rfc = self.data["radial_funcs_count"]
        rbs = self.data["radial_basis_size"]
        asm = self.data["alpha_scalar_moments"]

        print("#" * 75)
        print("scaling:", parameters[0])
        print("moment_coeffs:")
        print(parameters[1 : asm + 1])
        print("species_coeffs:")
        print(parameters[asm + 1 : asm + 1 + species_count])
        shape = species_count, species_count, rfc, rbs
        radial_coeffs = np.array(parameters[asm + 1 + species_count :]).reshape(shape)
        print("radial_coeffs:")
        print(radial_coeffs)
        print()
