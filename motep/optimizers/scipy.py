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
        if self.loss.comm.Get_rank() == 0:
            logger.info(f"loss {self.iter:4d}: {fun}")
            for handler in logger.handlers:
                handler.flush()
        self.iter += 1


class ScipyOptimizerBase(OptimizerBase):
    @property
    def optimized_default(self) -> list[str]:
        return ["species_coeffs", "moment_coeffs", "radial_coeffs"]

    @property
    def optimized_allowed(self) -> list[str]:
        return ["scaling", "species_coeffs", "moment_coeffs", "radial_coeffs"]

    def print_result(self, result: OptimizeResult) -> None:
        """Print `result`."""
        if self.loss.comm.Get_rank() == 0:
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


class ScipyDualAnnealingOptimizer(ScipyOptimizerBase):
    def optimize(self, **kwargs: dict[str, Any]) -> None:
        parameters = self.loss.mtp_data.parameters
        bounds = self.loss.mtp_data.get_bounds()
        callback = Callback(self.loss)
        callback(OptimizeResult(x=parameters, fun=self.loss(parameters)))
        result = dual_annealing(
            self.loss,
            bounds=bounds,
            callback=callback,
            seed=40,
            x0=parameters,
        )
        self.print_result(result)
        return result.x


class ScipyDifferentialEvolutionOptimizer(ScipyOptimizerBase):
    def optimize(self, **kwargs: dict[str, Any]) -> None:
        parameters = self.loss.mtp_data.parameters
        bounds = self.loss.mtp_data.get_bounds()
        callback = Callback(self.loss)
        callback(OptimizeResult(x=parameters, fun=self.loss(parameters)))
        result = differential_evolution(
            self.loss,
            bounds,
            popsize=30,
            callback=callback,
            x0=parameters,
        )
        self.print_result(result)
        return result.x


class ScipyMinimizeOptimizer(ScipyOptimizerBase):
    """`Optimizer` class using `scipy.minimize`."""

    def optimize(self, **kwargs: dict[str, Any]) -> None:
        """Optimizer using `scipy.optimize.minimize`."""
        parameters = self.loss.mtp_data.parameters
        bounds = self.loss.mtp_data.get_bounds()

        if kwargs.get("jac"):
            if "scaling" in self.optimized:
                raise ValueError("`jac` cannot (so far) be used to optimize `scaling`.")
            kwargs["jac"] = self.loss.jac
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
        callback = Callback(self.loss)
        callback(OptimizeResult(x=parameters, fun=self.loss(parameters)))
        result = minimize(
            self.loss,
            parameters,
            bounds=bounds,
            callback=callback,
            **kwargs,
        )
        self.print_result(result)
        self.loss.mtp_data.parameters = result.x
