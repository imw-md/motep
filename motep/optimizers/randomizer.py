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
        loss: LossFunctionBase,
        *,
        optimized: list[str] | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize `Randomizer`."""
        super().__init__(loss=loss, **kwargs)
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
        self.loss(parameters)
        rng: np.random.Generator = self.loss.setting["rng"]

        callback = Callback(self.loss)

        # Print the value of the loss function.
        callback(parameters)

        mtp_data = self.loss.mtp_data
        for key in self.optimized:
            lb = -5.0
            ub = +5.0
            shape = mtp_data[key].shape
            mtp_data[key] = rng.uniform(lb, ub, size=shape)

        parameters = mtp_data.parameters

        # Print the value of the loss function.
        callback(parameters)

        return parameters
