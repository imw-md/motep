"""Module for `Randomizer`."""

from typing import Any

import numpy as np

from motep.loss_function import LossFunctionBase
from motep.optimizers.base import OptimizerBase
from motep.optimizers.scipy import Callback


class Randomizer(OptimizerBase):
    """Special `Optimizer` class that actually randomizes parameters."""

    def __init__(
        self,
        loss_function: LossFunctionBase,
        optimized: list[str] | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize `Randomizer`."""
        super().__init__(loss_function)
        if optimized is None:
            optimized = ["species_coeffs", "radial_coeffs", "moment_coeffs"]
        self.optimized = optimized

    def optimize(
        self,
        parameters: np.ndarray,
        bounds: np.ndarray,
        **kwargs: dict[str, Any],
    ) -> np.ndarray:
        """Randomize parameters."""
        # Calculate basis functions of `fitness.images`
        self.loss_function(parameters)

        callback = Callback(self.loss_function)

        # Print the value of the loss function.
        callback(parameters)

        mtp_data = self.loss_function.mtp_data
        for key in self.optimized:
            lb = -5.0
            ub = +5.0
            shape = mtp_data.dict_mtp[key].shape
            mtp_data.dict_mtp[key] = mtp_data.rng.uniform(lb, ub, size=shape)

        parameters = mtp_data.parameters

        # Print the value of the loss function.
        callback(parameters)

        return parameters
