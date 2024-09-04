import random

from ase.calculators.morse import MorsePotential


def Morse_field(parameters):
    epsilon, rho0, r0, rcut1, rcut2 = parameters
    Potential = MorsePotential(
        epsilon=epsilon, rho0=rho0, r0=r0, rcut1=rcut1, rcut2=rcut2
    )
    return Potential


def generate_random_numbers(n, lower_limit, upper_limit, seed):
    random.seed(seed)
    random_numbers = [random.uniform(lower_limit, upper_limit) for _ in range(n)]
    return random_numbers
