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
        self.print_result(result)
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
        self.print_result(result)
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
            if "scaling" in self.optimized:
                raise ValueError("`jac` cannot (so far) be used to optimize `scaling`.")
            kwargs["jac"] = self.loss.jac
        callback = Callback(self.loss)
        result = minimize(
            self.loss,
            initial_guess,
            bounds=bounds,
            callback=callback,
            **kwargs,
        )
        self.print_result(result)
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
