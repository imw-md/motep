"""Module for the optimizer based on linear least squares (LLS)."""

import numpy as np

from motep.loss_function import LossFunction
from motep.opt import Callback


def optimization_lls(
    fitness: LossFunction,
    parameters: np.ndarray,
    bounds: np.ndarray,
) -> np.ndarray:
    """Optimize parameters.

    Parameters
    ----------
    fitness : :class:`~motep.loss_function.LossFunction`
        :class:`motep.loss_function.LossFunction` object.
    parameters : np.ndarray
        Initial parameters.
    bounds : np.ndarray
        Lower and upper bounds for the parameters.
        Not used in :class:`~motep.optimizers.lls.LLSOptimizer`.

    Returns
    -------
    parameters : np.ndarray
        Optimized parameters.

    """
    # Calculate basis functions of `fitness.images`
    fitness(parameters)

    energies = np.array(
        [atoms.get_potential_energy() for atoms in fitness.images],
    )

    basis_values = np.array(
        [atoms.calc.engine.basis_values for atoms in fitness.images],
    )

    # TODO: Consider also forces and stresses
    moment_coeffs = np.linalg.lstsq(basis_values, energies, rcond=None)[0]

    # TODO: Redesign optimizers to du such an assignment more semantically
    parameters[1 : len(moment_coeffs) + 1] = moment_coeffs

    # Print loss function value
    Callback(fitness)(parameters)

    return parameters
