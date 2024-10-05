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

from motep.optimizers import OptimizerBase


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
    print("  Final parameters:", result.x)
    print("  Final function value:", result.fun)


class ScipyDualAnnealingOptimizer(OptimizerBase):
    def optimize(
        self,
        initial_guess: np.ndarray,
        bounds: np.ndarray,
        **kwargs: dict[str, Any],
    ) -> np.ndarray:
        callback = Callback(self.loss_function)
        result = dual_annealing(
            self.loss_function,
            bounds=bounds,
            callback=callback,
            seed=40,
            x0=initial_guess,
        )
        print_result(result)
        return result.x


class ScipyDifferentialEvolutionOptimizer(OptimizerBase):
    def optimize(
        self,
        initial_guess: np.ndarray,
        bounds: np.ndarray,
        **kwargs: dict[str, Any],
    ) -> np.ndarray:
        callback = Callback(self.loss_function)
        result = differential_evolution(
            self.loss_function,
            bounds,
            popsize=30,
            callback=callback,
        )
        print_result(result)
        return result.x


class ScipyMinimizeOptimizer(OptimizerBase):
    """`Optimizer` class using `scipy.minimize`."""

    def optimize(
        self,
        initial_guess: np.ndarray,
        bounds: np.ndarray,
        **kwargs: dict[str, Any],
    ) -> np.ndarray:
        """Optimizer using `scipy.optimize.minimize`."""
        callback = Callback(self.loss_function)
        result = minimize(
            self.loss_function,
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
