"""Module for `Optimizer` classes."""

from motep.optimizers.base import OptimizerBase
from motep.optimizers.ga import GeneticAlgorithmOptimizer
from motep.optimizers.lls import LLSOptimizer
from motep.optimizers.scipy import (
    ScipyBFGSOptimizer,
    ScipyDifferentialEvolutionOptimizer,
    ScipyDualAnnealingOptimizer,
    ScipyMinimizeOptimizer,
    ScipyNelderMeadOptimizer,
)


def make_optimizer(optimizer: str) -> OptimizerBase:
    """Make an `Optimizer` class."""
    return {
        "GA": GeneticAlgorithmOptimizer,
        "minimize": ScipyMinimizeOptimizer,
        "Nelder-Mead": ScipyNelderMeadOptimizer,
        "L-BFGS-B": ScipyBFGSOptimizer,
        "DA": ScipyDualAnnealingOptimizer,
        "DE": ScipyDifferentialEvolutionOptimizer,
        "LLS": LLSOptimizer,
    }[optimizer]
