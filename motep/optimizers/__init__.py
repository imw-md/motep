"""Module for `Optimizer` classes."""

from motep.optimizers.base import OptimizerBase
from motep.optimizers.ga import GeneticAlgorithmOptimizer
from motep.optimizers.ideal import NoInteractionOptimizer
from motep.optimizers.level2mtp import Level2MTPOptimizer
from motep.optimizers.lls import LLSOptimizer
from motep.optimizers.randomizer import Randomizer
from motep.optimizers.scipy import (
    ScipyDifferentialEvolutionOptimizer,
    ScipyDualAnnealingOptimizer,
    ScipyLBFGSBOptimizer,
    ScipyMinimizeOptimizer,
    ScipyNelderMeadOptimizer,
)


def make_optimizer(optimizer: str) -> OptimizerBase:
    """Make an `Optimizer` class."""
    return {
        "GA": GeneticAlgorithmOptimizer,
        "NI": NoInteractionOptimizer,
        "Level2MTP": Level2MTPOptimizer,
        "minimize": ScipyMinimizeOptimizer,
        "Nelder-Mead": ScipyNelderMeadOptimizer,
        "L-BFGS-B": ScipyLBFGSBOptimizer,
        "DA": ScipyDualAnnealingOptimizer,
        "DE": ScipyDifferentialEvolutionOptimizer,
        "LLS": LLSOptimizer,
        "randomize": Randomizer,
    }[optimizer]
