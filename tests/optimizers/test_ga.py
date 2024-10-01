"""Tests for genetic algorithm (GA)."""

import numpy as np

from motep.optimizers.ga import GeneticAlgorithm, elite_callback


def _fun(x: np.ndarray) -> np.ndarray:
    n = len(x)
    sum1 = np.sum(np.square(x))
    sum2 = np.sum(np.cos(2 * np.pi * np.array(x)))
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e


def test_ga() -> None:
    """Test if `GeneticAlgorithm` runs."""
    population_size = 100
    initial_guess = [0.0, 0.0]
    mutation_rate = 0.1
    elitism_rate = 0.1
    crossover_probability = 0.9
    generations = 100
    lower_bound = [0.0] * len(initial_guess)
    upper_bound = [10.0] * len(initial_guess)

    ga = GeneticAlgorithm(
        _fun,
        initial_guess,
        population_size=population_size,
        mutation_rate=mutation_rate,
        elitism_rate=elitism_rate,
        crossover_probability=crossover_probability,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    ga.initialize_population()
    best_solution = ga.evolve_with_elites(_fun, generations, elite_callback)
    best_solution = ga.evolve_with_common(_fun, generations, elite_callback)
    best_solution = ga.evolve_with_mix(_fun, generations, elite_callback)
    best_solution = ga.evolve_with_steady(_fun, generations, elite_callback)
    print("Best solution found:", best_solution)
