from ase.calculators.morse import MorsePotential


def Morse_field(parameters):
    epsilon, rho0, r0, rcut1, rcut2 = parameters
    Potential = MorsePotential(
        epsilon=epsilon, rho0=rho0, r0=r0, rcut1=rcut1, rcut2=rcut2
    )
    return Potential
