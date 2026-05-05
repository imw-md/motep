"""Parsers of MLIP .mtp files."""

import itertools
import os
import warnings
from dataclasses import asdict
from numbers import Integral, Real
from pathlib import Path
from typing import TextIO

import numpy as np

from motep.potentials.mmtp.data import MagMTPData
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
        data["radial_basis"]["size"],
    )
    return np.array(coeffs).reshape(shape)


def _parse_radial_mag_coeffs(file: TextIO, data: dict) -> np.ndarray:
    coeffs = []

    rad_basis_size = data["radial_basis"]["size"]
    mag_basis_size = data["magnetic_basis"]["size"]

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
        rad_basis_size * mag_basis_size * mag_basis_size,
    )
    return np.array(coeffs).reshape(shape)


def _parse_mtp_file(file: os.PathLike) -> dict:
    """Parse an MLIP .mtp file into a raw data dict."""
    data: dict = {"radial_basis": {}, "magnetic_basis": {}}
    current_basis: dict | None = None

    with Path(file).open("r", encoding="utf-8") as fd:
        for line in fd:
            if line.strip() == "MTP":
                continue
            if "=" in line:
                key, value = (_.strip() for _ in line.strip().split("="))
                # Basis section headers
                if key == "radial_basis_type":
                    current_basis = data["radial_basis"]
                    current_basis["type"] = value.strip()
                elif key == "magnetic_basis_type":
                    current_basis = data["magnetic_basis"]
                    current_basis["type"] = value.strip()
                # Basis sub-keys (new format: min, max, size)
                elif key == "min" and current_basis is not None:
                    current_basis["min"] = np.float64(value)
                elif key == "max" and current_basis is not None:
                    current_basis["max"] = np.float64(value)
                elif key == "size" and current_basis is not None:
                    current_basis["size"] = np.int32(value)
                # Backward compat sub-keys (old format: min_val, max_val, basis_size)
                elif key == "min_val" and current_basis is not None:
                    current_basis["min"] = np.float64(value)
                elif key == "max_val" and current_basis is not None:
                    current_basis["max"] = np.float64(value)
                elif key == "basis_size" and current_basis is not None:
                    current_basis["size"] = np.int32(value)
                # Backward compat flat magnetic keys
                elif key == "min_mag":
                    data["magnetic_basis"]["min"] = np.float64(value)
                elif key == "max_mag":
                    data["magnetic_basis"]["max"] = np.float64(value)
                elif key == "mag_basis_size":
                    data["magnetic_basis"]["size"] = np.int32(value)
                # Backward compat flat radial keys
                elif key == "min_dist":
                    data["radial_basis"]["min"] = np.float64(value)
                elif key == "max_dist":
                    data["radial_basis"]["max"] = np.float64(value)
                elif key == "radial_basis_size":
                    data["radial_basis"]["size"] = np.int32(value)
                # Scalar fields
                elif key == "scaling":
                    data[key] = np.float64(value)
                elif value.isdigit():
                    data[key] = np.int32(value)
                elif key == "alpha_moment_mapping":
                    data[key] = np.fromiter(
                        (_ for _ in value.strip("{}").split(",")),
                        dtype=np.int32,
                    )
                elif key in {"species_coeffs", "moment_coeffs"}:
                    data[key] = np.fromiter(
                        (_ for _ in value.strip().strip("{}").split(",")),
                        dtype=np.float64,
                    )
                elif key in {"alpha_index_basic", "alpha_index_times"}:
                    data[key] = [
                        [int(_) for _ in _.split(",")]
                        for _ in value.strip("{}").split("}, {")
                        if _ != ""
                    ]
                    data[key] = np.array(data[key], dtype=np.int32)
                    # force to be two-dimensional even if empty (for Level 2)
                    if data[key].size == 0:
                        data[key] = np.zeros((0, 4), dtype=np.int32)
                else:
                    data[key] = value.strip()
            elif line.strip() == "radial_coeffs":
                if data["magnetic_basis"]:
                    data["radial_coeffs"] = _parse_radial_mag_coeffs(fd, data)
                else:
                    data["radial_coeffs"] = _parse_radial_coeffs(fd, data)

    return data


def read_mtp(file: os.PathLike) -> MTPData | MagMTPData:
    """Read an MLIP .mtp file, returning MagMTPData if a magnetic basis is present."""
    data = _parse_mtp_file(file)
    if data["magnetic_basis"]:
        return MagMTPData(**data)
    del data["magnetic_basis"]
    return MTPData(**data)


def read_mmtp(file: os.PathLike) -> MTPData | MagMTPData:
    """Read an MLIP .mtp file with magnetic basis."""
    return MagMTPData.from_base(read_mtp(file))


def _format_value(value: float | int | list | str) -> str:
    if isinstance(value, Integral):
        return f"{value:d}"
    if isinstance(value, Real):
        return f"{value:21.15e}"
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


def _write_basis_sections(fd: TextIO, data_dict: dict) -> None:
    for basis_key, type_key in [
        ("radial_basis", "radial_basis_type"),
        ("magnetic_basis", "magnetic_basis_type"),
    ]:
        basis = data_dict.get(basis_key, {})
        if basis:
            fd.write(f"{type_key} = {basis['type']}\n")
            fd.write(f"\tmin = {_format_value(basis['min'])}\n")
            fd.write(f"\tmax = {_format_value(basis['max'])}\n")
            fd.write(f"\tsize = {_format_value(basis['size'])}\n")


def _write_mtp_file(file: os.PathLike, data_dict: dict) -> None:
    """Write a raw data dict to an MLIP .mtp file."""
    keys0 = [
        "version",
        "potential_name",
        "scaling",
        "species_count",
        "potential_tag",
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
    species_count = data_dict["species_count"]
    with Path(file).open("w", encoding="utf-8") as fd:
        fd.write("MTP\n")
        for key in keys0:
            if data_dict.get(key) is not None:
                fd.write(f"{key} = {_format_value(data_dict[key])}\n")
        _write_basis_sections(fd, data_dict)
        if data_dict.get("radial_funcs_count") is not None:
            rfc = _format_value(data_dict["radial_funcs_count"])
            fd.write(f"radial_funcs_count = {rfc}\n")
        if data_dict.get("radial_coeffs") is not None:
            fd.write("\tradial_coeffs\n")
            for k0, k1 in itertools.product(range(species_count), repeat=2):
                value = data_dict["radial_coeffs"][k0, k1]
                fd.write(f"\t\t{k0}-{k1}\n")
                for _ in range(data_dict["radial_funcs_count"]):
                    fd.write(f"\t\t\t{_format_list(value[_])}\n")
        for key in keys2:
            if data_dict.get(key) is not None:
                fd.write(f"{key} = {_format_value(data_dict[key])}\n")


def write_mtp(
    file: os.PathLike,
    data: MTPData | MagMTPData,
    *,
    legacy: bool = True,
) -> None:
    """Write an MLIP .mtp file.

    Parameters
    ----------
    legacy : bool
        If True, write non-magnetic files using the old flat keyword format
        (``min_dist``, ``max_dist``, ``radial_basis_size``) instead of the
        grouped ``min`` / ``max`` / ``size`` section. Ignored for magnetic data.
    """
    if legacy and not isinstance(data, MagMTPData):
        _write_mtp_file_legacy(file, asdict(data))
    else:
        _write_mtp_file(file, asdict(data))


def _write_mtp_file_legacy(file: os.PathLike, data_dict: dict) -> None:
    """Write a non-magnetic MTP file using the old flat keyword format."""
    keys0 = [
        "version",
        "potential_name",
        "scaling",
        "species_count",
        "potential_tag",
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
    species_count = data_dict["species_count"]
    radial_basis = data_dict.get("radial_basis", {})
    with Path(file).open("w", encoding="utf-8") as fd:
        fd.write("MTP\n")
        for key in keys0:
            if data_dict.get(key) is not None:
                fd.write(f"{key} = {_format_value(data_dict[key])}\n")
        if radial_basis:
            fd.write(f"radial_basis_type = {radial_basis['type']}\n")
            fd.write(f"\tmin_dist = {_format_value(radial_basis['min'])}\n")
            fd.write(f"\tmax_dist = {_format_value(radial_basis['max'])}\n")
            fd.write(f"\tradial_basis_size = {_format_value(radial_basis['size'])}\n")
            fd.write(
                f"\tradial_funcs_count = {_format_value(data_dict['radial_funcs_count'])}\n"
            )
        if data_dict.get("radial_coeffs") is not None:
            fd.write("\tradial_coeffs\n")
            for k0, k1 in itertools.product(range(species_count), repeat=2):
                value = data_dict["radial_coeffs"][k0, k1]
                fd.write(f"\t\t{k0}-{k1}\n")
                fd.writelines(
                    f"\t\t\t{_format_list(value[_])}\n"
                    for _ in range(data_dict["radial_funcs_count"])
                )
        for key in keys2:
            if data_dict.get(key) is not None:
                fd.write(f"{key} = {_format_value(data_dict[key])}\n")


def write_mmtp(file: os.PathLike, data: MagMTPData) -> None:
    """Write an MLIP .mtp file with magnetic basis.

    .. deprecated::
        Use :func:`write_mtp` instead.
    """
    warnings.warn(
        "write_mmtp is deprecated, use write_mtp instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    write_mtp(file, data)
