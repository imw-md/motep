import random
from itertools import combinations_with_replacement

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


def read_untrained_MTP(file_path):
    # Read lines from the file
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Convert the text to YAML format
    yaml_data = {}
    for line in lines:
        d = line.strip().split(" = ")
        if len(d) > 1:
            key, value = d
            yaml_data[key] = value
        else:
            yaml_data[d[0]] = None

    return yaml_data


def brac(input):
    if len(input) == 1:
        return "{{{}}}".format(input[0])
    else:
        return "{{{}}}".format(",".join(map(str, input)))


def write_MTP(
    file,
    species_count,
    min_dist,
    max_dist,
    scaling,
    radial_coeffs,
    species_coeffs,
    moment_coeffs,
    yaml_data,
):
    version = yaml_data["version"]
    potential_name = yaml_data["potential_name"]
    potential_tag = ""
    radial_basis_type = yaml_data["radial_basis_type"]
    radial_basis_size = yaml_data["radial_basis_size"]
    radial_funcs_count = yaml_data["radial_funcs_count"]
    alpha_moments_count = yaml_data["alpha_moments_count"]
    alpha_index_basic_count = yaml_data["alpha_index_basic_count"]
    alpha_index_basic = yaml_data["alpha_index_basic"]
    alpha_index_times_count = yaml_data["alpha_index_times_count"]
    alpha_index_times = yaml_data["alpha_index_times"]
    alpha_scalar_moments = yaml_data["alpha_scalar_moments"]
    alpha_moment_mapping = yaml_data["alpha_moment_mapping"]

    with open(file, "w") as f:
        f.write("MTP\n")
        f.write("version = {}\n".format(version))
        f.write("potential_name = {}\n".format(potential_name))
        f.write("scaling = {}\n".format(scaling))
        f.write("species_count = {}\n".format(species_count))
        f.write("potential_tag = {}\n".format(potential_tag))
        f.write("radial_basis_type = {}\n".format(radial_basis_type))
        f.write("\tmin_dist = {:.2f}\n".format(min_dist))
        f.write("\tmax_dist = {:.2f}\n".format(max_dist))
        f.write("\tradial_basis_size = {}\n".format(radial_basis_size))
        f.write("\tradial_funcs_count = {}\n".format(radial_funcs_count))
        f.write("\tradial_coeffs\n")
        species_pairs = combinations_with_replacement(range(species_count), 2)
        j = 0
        for pair in species_pairs:
            f.write("\t\t{}-{}\n".format(pair[0], pair[1]))
            for m in range(int(radial_funcs_count)):
                # print((radial_coeffs[j]))
                f.write("\t\t\t{}\n".format(brac(radial_coeffs[j])))
                j += 1
        f.write("alpha_moments_count = {}\n".format(alpha_moments_count))
        f.write("alpha_index_basic_count = {}\n".format(alpha_index_basic_count))
        f.write("alpha_index_basic = {}\n".format(alpha_index_basic))
        f.write("alpha_index_times_count = {}\n".format(alpha_index_times_count))
        f.write("alpha_index_times = {}\n".format(alpha_index_times))
        f.write("alpha_scalar_moments = {}\n".format(alpha_scalar_moments))
        f.write("alpha_moment_mapping = {}\n".format(alpha_moment_mapping))
        f.write("species_coeffs = {}\n".format(brac(species_coeffs)))
        f.write("moment_coeffs = {}\n".format(brac(moment_coeffs)))
