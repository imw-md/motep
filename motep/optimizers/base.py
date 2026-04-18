"""Base class of the `Optimizer` classes."""

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt

from motep.loss import LossFunctionBase

logger = logging.getLogger(__name__)


class OptimizerBase(ABC):
    """Base class of the `Optimizer` classes.

    Attributes
    ----------
    loss : LossFunction
        :class:`motep.loss.LossFunction` object.
    optimized : list[str]
        Parameters to be optimized.

        - ``species_coeffs``
        - ``moment_coeffs``
        - ``radial_coeffs``

    """

    def __init__(
        self,
        loss: LossFunctionBase,
        **kwargs,
    ) -> None:
        """Initialize the `Optimizer` class.

        Parameters
        ----------
        loss : :class:`motep.loss.LossFunction`
            :class:`motep.loss.LossFunction` object.
        **kwargs
            Options passed to the `Optimizer` class.

        Raises
        ------
        ValueError

        """
        self.loss = loss

        optimized: list[str] | None = kwargs.get("optimized")
        if optimized is None:
            optimized = self.optimized_default.copy()
        if not all(_ in self.optimized_allowed for _ in optimized):
            msg = f"Some keywords cannot be optimized with {__name__}."
            raise ValueError(msg)

        # add always optimized parameters
        self.optimized = optimized + self.optimized_always

        # avoid duplication of parameters
        self.optimized = sorted(set(self.optimized), key=self.optimized.index)

        self.loss.mtp_data.optimized = self.optimized

    @abstractmethod
    def optimize(self, **kwargs: dict[str, Any]) -> None:
        """Run the optimizer."""

    @property
    @abstractmethod
    def optimized_default(self) -> list[str]:
        """Return default `optimized`."""

    @property
    @abstractmethod
    def optimized_allowed(self) -> list[str]:
        """Return allowed `optimized`."""

    @property
    def optimized_always(self) -> list[str]:
        """Parameters optimized always regardless they are specified or not."""
        return []


class ParallelOptimizerBase(OptimizerBase):
    """OptimizerBase with master-worker MPI pattern.

    Only rank 0 runs the optimizer logic (in :meth:`_optimize`);
    other ranks service the collective MPI operations triggered by
    each loss / jac / gather_data evaluation.
    """

    _OP_LOSS = 0
    _OP_JAC = 1
    _OP_GATHER = 2
    _OP_STOP = 3

    def rank0_loss(self, parameters: npt.NDArray[np.float64]) -> float:
        """Evaluate the loss function, signaling workers first.

        Returns
        -------
        float

        """
        self.loss.comm.bcast(self._OP_LOSS, root=0)
        return self.loss(parameters)

    def rank0_jac(
        self,
        parameters: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Evaluate the Jacobian, signaling workers first.

        Returns
        -------
        npt.NDArray[np.float64]

        """
        self.loss.comm.bcast(self._OP_JAC, root=0)
        return self.loss.jac(parameters)

    def rank0_gather_data(self) -> None:
        """Gather data to rank 0, signaling workers first."""
        self.loss.comm.bcast(self._OP_GATHER, root=0)
        self.loss.gather_data()

    def _worker_loop(self) -> None:
        """Service collective operations from rank 0's optimizer.

        Only rank 0 runs the optimizer; other ranks call this method
        to participate in the collective MPI operations (bcast, allreduce)
        triggered by each loss/jac/gather evaluation.
        """
        while True:
            op = self.loss.comm.bcast(None, root=0)
            if op == self._OP_STOP:
                break
            elif op == self._OP_LOSS:
                self.loss(None)
            elif op == self._OP_JAC:
                self.loss.jac(None)
            elif op == self._OP_GATHER:
                self.loss.gather_data()

    def optimize(self, **kwargs: dict[str, Any]) -> None:
        """Run the optimizer with the master-worker pattern.

        Rank 0 runs :meth:`_optimize`; other ranks service collective MPI
        operations in a loop.  After :meth:`_optimize` returns, the
        resulting parameters are broadcast to all ranks.
        """
        if self.loss.comm.rank == 0:
            result_x = self._optimize(**kwargs)
            self.loss.comm.bcast(self._OP_STOP, root=0)
        else:
            self._worker_loop()
            result_x = None
        result_x = self.loss.comm.bcast(result_x, root=0)
        self.loss.mtp_data.parameters = result_x

    @abstractmethod
    def _optimize(self, **kwargs: dict[str, Any]) -> npt.NDArray[np.float64]:
        """Run the optimizer on rank 0.

        Must use :meth:`rank0_loss`, :meth:`rank0_jac`, and
        :meth:`rank0_gather_data` instead of calling ``self.loss`` directly,
        so that worker ranks participate in the collective MPI operations.

        Returns
        -------
        parameters : npt.NDArray[np.float64]
            Optimized parameters.

        """
