"""Base class of the `Optimizer` classes."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from motep.loss_function import LossFunctionBase


class OptimizerBase(ABC):
    """Base class of the `Optimizer` classes.

    Attributes
    ----------
    mtp_data : MTPData
        :class:`motep.potentials.MTPData` object.

    """

    def __init__(self, loss: LossFunctionBase, **kwargs: dict[str, Any]) -> None:
        """Initialize the `Optimizer` class.

        Parameters
        ----------
        loss : :class:`motep.loss_function.LossFunction`
            :class:`motep.loss_function.LossFunction` object.
        **kwargs : dict[str, Any]
            Options passed to the `Optimizer` class.

        """
        self.loss = loss
        mtp_data = self.loss.mtp_data
        if "species" not in mtp_data:
            species = {_: _ for _ in range(mtp_data["species_count"])}
            mtp_data["species"] = species

    @abstractmethod
    def optimize(
        self,
        parameters: np.ndarray,
        bounds: np.ndarray,
        **kwargs: dict[str, Any],
    ) -> np.ndarray:
        """Optimize parameters.

        Parameters
        ----------
        parameters : np.ndarray
            Initial parameters.
        bounds : np.ndarray
            Lower and upper bounds for the parameters.
        **kwargs : dict[str, Any]
            Other keyward arguments passed to the actual optimization function.

        Returns
        -------
        parameters : np.ndarray
            Optimized parameters.

        """
