"""Functions related to the setting file."""

import pathlib
import tomllib
from typing import Any


def make_default_setting() -> dict[str, Any]:
    """Make default setting."""
    # The keys are set one by one here to fix their order.
    setting = {}
    setting["configurations"] = "training.cfg"
    setting["potential_initial"] = "initial.mtp"
    setting["potential_final"] = "final.mtp"
    setting["seed"] = None
    setting["engine"] = "numpy"
    setting["energy-weight"] = 1.0
    setting["force-weight"] = 0.01
    setting["stress-weight"] = 0.0
    setting["steps"] = [
        {"method": "L-BFGS-B", "optimized": ["radial_coeffs", "moment_coeffs"]},
    ]
    return setting


def parse_setting(filename: str) -> dict:
    """Parse setting file."""
    with pathlib.Path(filename).open("rb") as f:
        setting = tomllib.load(f)

    # convert the old style "steps" like {'steps`: ['L-BFGS-B']} to the new one
    # {'steps`: {'method': 'L-BFGS-B', 'optimized': ['radial_coeffs', 'moment_coeffs']}
    optimized_default = [
        "scaling",
        "species_coeffs",
        "radial_coeffs",
        "moment_coeffs",
    ]
    for i, value in enumerate(setting["steps"]):
        if not isinstance(value, dict):
            setting["steps"][i] = {
                "method": value,
                "optimized": optimized_default,
            }

    return setting
