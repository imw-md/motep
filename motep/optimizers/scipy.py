"""Optimizers based on SciPy."""

import logging
from typing import Any

from scipy.optimize import (
    OptimizeResult,
    differential_evolution,
    dual_annealing,
    minimize,
)

from motep.loss import LossFunctionBase
from motep.optimizers.base import OptimizerBase

logger = logging.getLogger(__name__)


class Callback:
    """Callback after each iteration."""

    def __init__(self, loss: LossFunctionBase):
        self.loss = loss
        self.iter = 0

    def __call__(self, intermediate_result: OptimizeResult):
        fun = intermediate_result.fun
        if self.loss.comm.rank == 0:
            logger.info(f"loss {self.iter:4d}: {fun}")
            for handler in logger.handlers:
                handler.flush()
        self.iter += 1


class ScipyOptimizerBase(OptimizerBase):
    _OP_LOSS = 0
    _OP_JAC = 1
    _OP_STOP = 2

    @property
    def optimized_default(self) -> list[str]:
        return ["species_coeffs", "moment_coeffs", "radial_coeffs"]

    @property
    def optimized_allowed(self) -> list[str]:
        return ["scaling", "species_coeffs", "moment_coeffs", "radial_coeffs"]

    def _rank0_loss(self, parameters):
        """Loss wrapper for rank 0 — signals workers before evaluation."""
        self.loss.comm.bcast(self._OP_LOSS, root=0)
        return self.loss(parameters)

    def _rank0_jac(self, parameters):
        """Jac wrapper for rank 0 — signals workers before evaluation."""
        self.loss.comm.bcast(self._OP_JAC, root=0)
        return self.loss.jac(parameters)

    def _worker_loop(self):
        """Service collective loss/jac evaluations from rank 0's optimizer.

        Only rank 0 runs scipy's optimizer; other ranks call this method
        to participate in the collective MPI operations (bcast, allreduce)
        triggered by each loss/jac evaluation.
        """
        while True:
            op = self.loss.comm.bcast(None, root=0)
            if op == self._OP_STOP:
                break
            elif op == self._OP_LOSS:
                self.loss(None)
            elif op == self._OP_JAC:
                self.loss.jac(None)

    def print_result(self, result: OptimizeResult) -> None:
        """Print `result`."""
        if self.loss.comm.rank == 0:
            logger.info("")
            for handler in logger.handlers:
                handler.flush()
            logger.info(f"Optimization result:")
            logger.info(f"  Message: {result.message}")
            logger.info(f"  Success: {result.success}")
            logger.info(f"  Status code: {result.status}")
            logger.info(f"  Number of function evaluations: {result.nfev}")
            logger.info(f"  Number of iterations: {result.nit}")
            # logger.info(f"  Final parameters: {result.x}")
            # logger.info(f"  Final function value: {result.fun}")

    def optimize(self) -> None:
        """Optimize with a `scipy.optimize` function."""
        p = self.loss.mtp_data.parameters
        self.callback = Callback(self.loss)
        self.callback(OptimizeResult(x=p, fun=self.loss(p)))


class ScipyDualAnnealingOptimizer(ScipyOptimizerBase):
    def optimize(self, **kwargs: dict[str, Any]) -> None:
        super().optimize()
        parameters = self.loss.mtp_data.parameters
        bounds = self.loss.mtp_data.get_bounds()
        if self.loss.comm.rank == 0:
            result = dual_annealing(
                self._rank0_loss,
                bounds=bounds,
                callback=self.callback,
                seed=40,
                x0=parameters,
            )
            self.loss.comm.bcast(self._OP_STOP, root=0)
            self.print_result(result)
        else:
            result = OptimizeResult(x=None)
            self._worker_loop()
        result.x = self.loss.comm.bcast(result.x, root=0)
        self.loss.mtp_data.parameters = result.x


class ScipyDifferentialEvolutionOptimizer(ScipyOptimizerBase):
    def optimize(self, **kwargs: dict[str, Any]) -> None:
        super().optimize()
        parameters = self.loss.mtp_data.parameters
        bounds = self.loss.mtp_data.get_bounds()
        if self.loss.comm.rank == 0:
            result = differential_evolution(
                self._rank0_loss,
                bounds,
                popsize=30,
                callback=self.callback,
                x0=parameters,
            )
            self.loss.comm.bcast(self._OP_STOP, root=0)
            self.print_result(result)
        else:
            result = OptimizeResult(x=None)
            self._worker_loop()
        result.x = self.loss.comm.bcast(result.x, root=0)
        self.loss.mtp_data.parameters = result.x


class ScipyMinimizeOptimizer(ScipyOptimizerBase):
    """`Optimizer` class using `scipy.minimize`."""

    def optimize(self, **kwargs: dict[str, Any]) -> None:
        """Optimizer using `scipy.optimize.minimize`."""
        super().optimize()
        parameters = self.loss.mtp_data.parameters
        bounds = self.loss.mtp_data.get_bounds()

        if kwargs.get("jac"):
            if "scaling" in self.optimized:
                raise ValueError("`jac` cannot (so far) be used to optimize `scaling`.")
            kwargs["jac"] = self._rank0_jac
        if kwargs.get("method", "").lower() in {
            "cg",
            "bfgs",
            "newton-cg",
            "dogleg",
            "trust-ncg",
            "trust-exact",
            "trust-krylov",
        }:
            bounds = None

        if self.loss.comm.rank == 0:
            result = minimize(
                self._rank0_loss,
                parameters,
                bounds=bounds,
                callback=self.callback,
                **kwargs,
            )
            self.loss.comm.bcast(self._OP_STOP, root=0)
            self.print_result(result)
        else:
            result = OptimizeResult(x=None)
            self._worker_loop()
        result.x = self.loss.comm.bcast(result.x, root=0)
        self.loss.mtp_data.parameters = result.x
