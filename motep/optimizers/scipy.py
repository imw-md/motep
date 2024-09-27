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


def optimization_sa(fun, initial_guess, bounds, **kwargs) -> np.ndarray:
    callback = Callback(fun)
    result = dual_annealing(
        fun,
        bounds=bounds,
        callback=callback,
        seed=40,
        x0=initial_guess,
    )
    print_result(result)
    return result.x


def optimize_minimize(
    fun: Callable,
    initial_guess: np.ndarray,
    bounds: np.ndarray,
    **kwargs: dict[str, Any],
) -> np.ndarray:
    """Optimizer using `scipy.optimize.minimize`."""
    callback = Callback(fun)
    result = minimize(
        fun,
        initial_guess,
        bounds=bounds,
        callback=callback,
        **kwargs,
    )
    print_result(result)
    return result.x


def optimization_nelder(fun, initial_guess, bounds, **kwargs) -> np.ndarray:
    return optimize_minimize(
        fun,
        initial_guess,
        bounds,
        method="Nelder-Mead",
        **kwargs,
    )


def optimization_bfgs(fun, initial_guess, bounds, **kwargs) -> np.ndarray:
    return optimize_minimize(
        fun,
        initial_guess,
        bounds,
        method="L-BFGS-B",
        **kwargs,
    )


def optimization_DE(fun, initial_guess, bounds, **kwargs) -> np.ndarray:
    callback = Callback(fun)
    result = differential_evolution(
        fun,
        bounds,
        popsize=30,
        callback=callback,
    )
    print_result(result)
    return result.x
