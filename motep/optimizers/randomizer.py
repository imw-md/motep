"""Module for `Randomizer`."""

from typing import Any

import numpy as np
from scipy.optimize._optimize import OptimizeResult

from motep.loss import LossFunctionBase
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

    def optimize(self, **kwargs: dict[str, Any]) -> None:
        parameters = self.loss.mtp_data.parameters
        callback = Callback(self.loss)
        rng: np.random.Generator = self.loss.setting["rng"]

        # Calculate basis functions of `loss.images`
        loss_value = self.loss(parameters)
        self.loss.gather_data()

        # Print the value of the loss function.
        callback(OptimizeResult(x=parameters, fun=loss_value))

        mtp_data = self.loss.mtp_data
        for key in self.optimized:
            lb = -5.0
            ub = +5.0
            shape = mtp_data[key].shape
            mtp_data[key] = rng.uniform(lb, ub, size=shape)
        # Update `parameters` by calling the property
        parameters = mtp_data.parameters

        # Evaluate loss with the new parameters
        loss_value = self.loss(parameters)

        # Print the value of the loss function.
        callback(OptimizeResult(x=parameters, fun=loss_value))
