"""Optimizers based on SciPy."""

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import (
    OptimizeResult,
    differential_evolution,
    dual_annealing,
    minimize,
)

from motep.optimizers.base import OptimizerBase


class Callback:
    """Callback after each iteration."""

    def __init__(self, fun: Callable):
        self.fun = fun

    def __call__(self, intermediate_result: OptimizeResult | np.ndarray):
        fun = (
            intermediate_result.fun
            if isinstance(intermediate_result, OptimizeResult)
            else self.fun(intermediate_result)
        )
        print("Function value:", fun)


def print_result(result: OptimizeResult) -> None:
    """Print `result`."""
    print("Optimization result:")
    print("  Message:", result.message)
    print("  Success:", result.success)
    print("  Status code:", result.status)
    print("  Number of function evaluations:", result.nfev)
    print("  Number of iterations:", result.nit)
    # print("  Final parameters:", result.x)
    # print("  Final function value:", result.fun)


class ScipyOptimizerBase(OptimizerBase):
    @property
    def optimized_default(self) -> list[str]:
        return ["species_coeffs", "moment_coeffs", "radial_coeffs"]

    @property
    def optimized_allowed(self) -> list[str]:
        return ["scaling", "species_coeffs", "moment_coeffs", "radial_coeffs"]


class ScipyDualAnnealingOptimizer(ScipyOptimizerBase):
    def optimize(
        self,
        initial_guess: np.ndarray,
        bounds: np.ndarray,
        **kwargs: dict[str, Any],
    ) -> np.ndarray:
        callback = Callback(self.loss)
        result = dual_annealing(
            self.loss,
            bounds=bounds,
            callback=callback,
            seed=40,
            x0=initial_guess,
        )
        print_result(result)
        return result.x


class ScipyDifferentialEvolutionOptimizer(ScipyOptimizerBase):
    def optimize(
        self,
        initial_guess: np.ndarray,
        bounds: np.ndarray,
        **kwargs: dict[str, Any],
    ) -> np.ndarray:
        callback = Callback(self.loss)
        result = differential_evolution(
            self.loss,
            bounds,
            popsize=30,
            callback=callback,
        )
        print_result(result)
        return result.x


class ScipyMinimizeOptimizer(ScipyOptimizerBase):
    """`Optimizer` class using `scipy.minimize`."""

    def optimize(
        self,
        initial_guess: np.ndarray,
        bounds: np.ndarray,
        **kwargs: dict[str, Any],
    ) -> np.ndarray:
        """Optimizer using `scipy.optimize.minimize`."""
        if kwargs.get("jac"):
            kwargs["jac"] = self.loss.jac
        callback = Callback(self.loss)
        result = minimize(
            self.loss,
            initial_guess,
            bounds=bounds,
            callback=callback,
            **kwargs,
        )
        print_result(result)
        return result.x


class ScipyNelderMeadOptimizer(ScipyMinimizeOptimizer):
    def optimize(
        self,
        initial_guess: np.ndarray,
        bounds: np.ndarray,
        **kwargs: dict[str, Any],
    ) -> np.ndarray:
        return super().optimize(
            initial_guess,
            bounds,
            method="Nelder-Mead",
            **kwargs,
        )


class ScipyBFGSOptimizer(ScipyMinimizeOptimizer):
    def optimize(
        self,
        initial_guess: np.ndarray,
        bounds: np.ndarray,
        **kwargs: dict[str, Any],
    ) -> np.ndarray:
        return super().optimize(
            initial_guess,
            bounds,
            method="L-BFGS-B",
            **kwargs,
        )
