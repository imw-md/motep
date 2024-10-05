"""Base class of the `Optimizer` classes."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np

from motep.potentials import MTPData


class OptimizerBase(ABC):
    """Base class of the `Optimizer` classes.

    Attributes
    ----------
    mtp_data : MTPData
        :class:`motep.potentials.MTPData` object.

    """

    def __init__(self, mtp_data: MTPData) -> None:
        """Initialize the `Optimizer` class."""
        self.mtp_data = mtp_data
        if "species" not in self.mtp_data.dict_mtp:
            species = {_: _ for _ in range(self.mtp_data.dict_mtp["species_count"])}
            self.mtp_data.dict_mtp["species"] = species

    @abstractmethod
    def optimize(
        self,
        fun: Callable,
        parameters: np.ndarray,
        bounds: np.ndarray,
        **kwargs: dict[str, Any],
    ) -> np.ndarray:
        """Optimize parameters.

        Parameters
        ----------
        fun : Callable
            Callable returing the value of the loss function.
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
