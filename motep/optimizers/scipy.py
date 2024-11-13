"""Optimizers based on SciPy."""

from typing import Any

import numpy as np
from scipy.optimize import (
    OptimizeResult,
    differential_evolution,
    dual_annealing,
    minimize,
)

from motep.loss import LossFunctionBase
from motep.optimizers.base import OptimizerBase


class Callback:
    """Callback after each iteration."""

    def __init__(self, loss: LossFunctionBase):
        self.loss = loss
        self.iter = 0

    def __call__(self, intermediate_result: OptimizeResult | np.ndarray):
        fun = (
            intermediate_result.fun
            if isinstance(intermediate_result, OptimizeResult)
            else self.loss(intermediate_result)
        )
        if self.loss.comm.Get_rank() == 0:
            print(f"loss {self.iter:4d}:", fun)
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
            print("Optimization result:")
            print("  Message:", result.message)
            print("  Success:", result.success)
            print("  Status code:", result.status)
            print("  Number of function evaluations:", result.nfev)
            print("  Number of iterations:", result.nit)
            # print("  Final parameters:", result.x)
            # print("  Final function value:", result.fun)


class ScipyDualAnnealingOptimizer(ScipyOptimizerBase):
    def optimize(self, **kwargs: dict[str, Any]) -> None:
        parameters = self.loss.mtp_data.parameters
        bounds = self.loss.mtp_data.get_bounds()
        callback = Callback(self.loss)
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
        if kwargs["method"].lower() not in {
            "nelder-mead",
            "powell",
            "l-bfgs-b",
            "cobyla",
            "cobyqa",
            "slsqp",
            "tnc",
            "trust-constr",
            "_custom",
        }:
            bounds = None
        callback = Callback(self.loss)
        result = minimize(
            self.loss,
            parameters,
            bounds=bounds,
            callback=callback,
            **kwargs,
        )
        self.print_result(result)
        self.loss.mtp_data.parameters = result.x
