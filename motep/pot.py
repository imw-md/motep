import random
from itertools import combinations_with_replacement
from typing import Any

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


def read_untrained_MTP(file_path) -> dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    data = {}
    for line in lines:
        d = line.strip().split(" = ")
        if len(d) > 1:
            key, value = d
            if value.isdigit():
                data[key] = int(value)
            elif key in ["scaling", "min_dist", "max_dist"]:
                data[key] = float(value)
            else:
                data[key] = value
        else:
            data[d[0]] = None

    return data


def brac(input):
    if len(input) == 1:
        return "{{{}}}".format(input[0])
    else:
        return "{{{}}}".format(",".join(map(str, input)))


def write_MTP(
    file,
    scaling,
    radial_coeffs,
    species_coeffs,
    moment_coeffs,
    data: dict[str, Any],
):
    version = data["version"]
    potential_name = data["potential_name"]
    species_count = data["species_count"]
    potential_tag = ""
    radial_basis_type = data["radial_basis_type"]
    radial_basis_size = data["radial_basis_size"]
    radial_funcs_count = data["radial_funcs_count"]
    alpha_moments_count = data["alpha_moments_count"]
    alpha_index_basic_count = data["alpha_index_basic_count"]
    alpha_index_basic = data["alpha_index_basic"]
    alpha_index_times_count = data["alpha_index_times_count"]
    alpha_index_times = data["alpha_index_times"]
    alpha_scalar_moments = data["alpha_scalar_moments"]
    alpha_moment_mapping = data["alpha_moment_mapping"]

    with open(file, "w", encoding="utf-8") as f:
        f.write("MTP\n")
        f.write(f"version = {version}\n")
        f.write(f"potential_name = {potential_name}\n")
        f.write(f"scaling = {scaling:21.15e}\n")
        f.write(f"species_count = {species_count:d}\n")
        f.write(f"potential_tag = {potential_tag}\n")
        f.write(f"radial_basis_type = {radial_basis_type}\n")
        f.write(f"\tmin_dist = {data['min_dist']:21.15e}\n")
        f.write(f"\tmax_dist = {data['max_dist']:21.15e}\n")
        f.write(f"\tradial_basis_size = {radial_basis_size:d}\n")
        f.write(f"\tradial_funcs_count = {radial_funcs_count:d}\n")
        f.write("\tradial_coeffs\n")
        species_pairs = combinations_with_replacement(range(species_count), 2)
        j = 0
        for pair in species_pairs:
            f.write(f"\t\t{pair[0]}-{pair[1]}\n")
            for _ in range(int(radial_funcs_count)):
                f.write("\t\t\t{}\n".format(brac(radial_coeffs[j])))
                j += 1
        f.write(f"alpha_moments_count = {alpha_moments_count}\n")
        f.write(f"alpha_index_basic_count = {alpha_index_basic_count}\n")
        f.write(f"alpha_index_basic = {alpha_index_basic}\n")
        f.write(f"alpha_index_times_count = {alpha_index_times_count}\n")
        f.write(f"alpha_index_times = {alpha_index_times}\n")
        f.write(f"alpha_scalar_moments = {alpha_scalar_moments}\n")
        f.write(f"alpha_moment_mapping = {alpha_moment_mapping}\n")
        f.write("species_coeffs = {}\n".format(brac(species_coeffs)))
        f.write("moment_coeffs = {}\n".format(brac(moment_coeffs)))
