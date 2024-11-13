"""Base class of the `Optimizer` classes."""

from abc import ABC, abstractmethod
from typing import Any

from motep.loss import LossFunctionBase


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
        loss : :class:`motep.loss.LossFunction`
            :class:`motep.loss.LossFunction` object.
        **kwargs : dict[str, Any]
            Options passed to the `Optimizer` class.

        """
        self.loss = loss

        if "optimized" not in kwargs:
            self.optimized = self.optimized_default
        elif all(_ in self.optimized_allowed for _ in kwargs["optimized"]):
            self.optimized = kwargs["optimized"]
        else:
            msg = f"Some keywords cannot be optimized in {__name__}."
            raise ValueError(msg)

        self.loss.mtp_data.optimized = self.optimized

    @abstractmethod
    def optimize(self, **kwargs: dict[str, Any]) -> None:
        """Optimize parameters."""

    @property
    @abstractmethod
    def optimized_default(self) -> list[str]:
        """Return default `optimized`."""

    @property
    @abstractmethod
    def optimized_allowed(self) -> list[str]:
        """Return allowed `optimized`."""
