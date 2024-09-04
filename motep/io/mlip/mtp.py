from typing import Any, TextIO

import numpy as np


def _parse_radial_coeffs(
    file: TextIO,
    data: dict[str, Any],
) -> dict[tuple[int, int], list[list[float]]]:
    d = {}
    for _ in range(data["species_count"]):
        for _ in range(data["species_count"]):
            key = tuple(int(_) for _ in next(file).strip().split("-"))
            value = []
            for _ in range(data["radial_funcs_count"]):
                tmp = next(file).strip().strip("{}").split(",")
                value.append([float(_) for _ in tmp])
            d[key] = value
    return d


def read_mtp(file_path) -> dict[str, Any]:
    data = {}
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip() == "MTP":
                continue
            if "=" in line:
                key, value = [_.strip() for _ in line.strip().split("=")]
                if key in ["scaling", "min_dist", "max_dist"]:
                    data[key] = float(value)
                elif value.isdigit():
                    data[key] = int(value)
                elif key in ["alpha_moment_mapping"]:
                    data[key] = [int(_) for _ in value.strip("{}").split(",")]
                elif key in ["species_coeffs", "moment_coeffs"]:
                    data[key] = [float(_) for _ in value.strip().strip("{}").split(",")]
                elif key in ["alpha_index_basic", "alpha_index_times"]:
                    data[key] = [
                        [int(_) for _ in _.split(",")]
                        for _ in value.strip("{}").split("}, {")
                        if _ != ""
                    ]
                else:
                    data[key] = value.strip()
            elif line.strip() == "radial_coeffs":
                key = "radial_coeffs"
                data[key] = _parse_radial_coeffs(file, data)

    return data


def _format_value(value: float | int | list | str) -> str:
    if isinstance(value, float):
        return f"{value:21.15e}"
    if isinstance(value, int):
        return f"{value:d}"
    if isinstance(value, list):
        return _format_list(value)
    if isinstance(value, np.ndarray):
        return _format_list(value.tolist())
    return value.strip()


def _format_list(value: list) -> str:
    if len(value) == 0:
        return "{}"
    if isinstance(value[0], list):
        return "{" + ", ".join(_format_list(_) for _ in value) + "}"
    return "{" + ", ".join(f"{_format_value(_)}" for _ in value) + "}"


def write_mtp(file, data: dict[str, Any]) -> None:
    keys0 = [
        "version",
        "potential_name",
        "scaling",
        "species_count",
        "potential_tag",
        "radial_basis_type",
    ]
    keys1 = [
        "min_dist",
        "max_dist",
        "radial_basis_size",
        "radial_funcs_count",
    ]
    keys2 = [
        "alpha_moments_count",
        "alpha_index_basic_count",
        "alpha_index_basic",
        "alpha_index_times_count",
        "alpha_index_times",
        "alpha_scalar_moments",
        "alpha_moment_mapping",
        "species_coeffs",
        "moment_coeffs",
    ]

    with open(file, "w", encoding="utf-8") as fd:
        fd.write("MTP\n")
        for key in keys0:
            if key in data:
                fd.write(f"{key} = {_format_value(data[key])}\n")
        for key in keys1:
            if key in data:
                fd.write(f"\t{key} = {_format_value(data[key])}\n")
        if "radial_coeffs" in data:
            fd.write("\tradial_coeffs\n")
            for key, value in data["radial_coeffs"].items():
                fd.write(f"\t\t{key[0]}-{key[1]}\n")
                for _ in range(data["radial_funcs_count"]):
                    fd.write(f"\t\t\t{_format_list(value[_])}\n")
        for key in keys2:
            if key in data:
                fd.write(f"{key} = {_format_value(data[key])}\n")
