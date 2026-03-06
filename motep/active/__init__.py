"""Module for active learning."""

from .algorithms import AlgorithmBase, ExhaustiveAlgorithm, MaxVolAlgorithm


def make_algorithm(algorithm: str) -> AlgorithmBase:
    """Make an `Algorithm` class."""
    return {
        "exhaustive": ExhaustiveAlgorithm,
        "maxvol": MaxVolAlgorithm,
    }[algorithm]
