"""Parsers of MLIP .mtp files."""

import itertools
import os
import pathlib
from dataclasses import asdict
from typing import TextIO

import numpy as np

from motep.potentials.mtp.data import MTPData


def _parse_radial_coeffs(file: TextIO, data: dict) -> np.ndarray:
    coeffs = []
    for _ in range(data["species_count"]):
        for _ in range(data["species_count"]):
            next(file)  # skip line with e.g. `0-0`
            for _ in range(data["radial_funcs_count"]):
                tmp = next(file).strip().strip("{}").split(",")
                coeffs.append([float(_) for _ in tmp])
    shape = (
        data["species_count"],
        data["species_count"],
        data["radial_funcs_count"],
        data["radial_basis_size"],
    )
    return np.array(coeffs).reshape(shape)


def read_mtp(file: os.PathLike) -> MTPData:
    """Read an MLIP .mtp file."""
    data = {}
    with pathlib.Path(file).open("r", encoding="utf-8") as fd:
        for line in fd:
            if line.strip() == "MTP":
                continue
            if "=" in line:
                key, value = (_.strip() for _ in line.strip().split("="))
                if key in {"scaling", "min_dist", "max_dist"}:
                    data[key] = float(value)
                elif value.isdigit():
                    data[key] = int(value)
                elif key == "alpha_moment_mapping":
                    data[key] = np.fromiter(
                        (_ for _ in value.strip("{}").split(",")),
                        dtype=int,
                    )
                elif key in {"species_coeffs", "moment_coeffs"}:
                    data[key] = np.fromiter(
                        (_ for _ in value.strip().strip("{}").split(",")),
                        dtype=float,
                    )
                elif key in {"alpha_index_basic", "alpha_index_times"}:
                    data[key] = [
                        [int(_) for _ in _.split(",")]
                        for _ in value.strip("{}").split("}, {")
                        if _ != ""
                    ]
                    data[key] = np.array(data[key])
                    # force to be two-dimensional even if empty (for Level 2)
                    if data[key].size == 0:
                        data[key] = np.zeros((0, 4), dtype=int)
                else:
                    data[key] = value.strip()
            elif line.strip() == "radial_coeffs":
                key = "radial_coeffs"
                data[key] = _parse_radial_coeffs(fd, data)

    return MTPData(**data)


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


def write_mtp(file: os.PathLike, data: MTPData) -> None:
    """Write an MLIP .mtp file."""
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
    data = asdict(data)
    species_count = data["species_count"]
    with pathlib.Path(file).open("w", encoding="utf-8") as fd:
        fd.write("MTP\n")
        for key in keys0:
            if data[key] is not None:
                fd.write(f"{key} = {_format_value(data[key])}\n")
        for key in keys1:
            if data[key] is not None:
                fd.write(f"\t{key} = {_format_value(data[key])}\n")
        if data["radial_coeffs"] is not None:
            fd.write("\tradial_coeffs\n")
            for k0, k1 in itertools.product(range(species_count), repeat=2):
                value = data["radial_coeffs"][k0, k1]
                fd.write(f"\t\t{k0}-{k1}\n")
                for _ in range(data["radial_funcs_count"]):
                    fd.write(f"\t\t\t{_format_list(value[_])}\n")
        for key in keys2:
            if data[key] is not None:
                fd.write(f"{key} = {_format_value(data[key])}\n")
