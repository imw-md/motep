"""Module for active learning."""

from .algorithms import AlgorithmBase, ExaustiveAlgorithm, MaxVolAlgorithm


def make_algorithm(algorithm: str) -> AlgorithmBase:
    """Make an `Algorithm` class."""
    return {
        "exaustive": ExaustiveAlgorithm,
        "maxvol": MaxVolAlgorithm,
    }[algorithm]
