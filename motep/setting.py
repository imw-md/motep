"""Functions related to the setting file."""

import pathlib
import tomllib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Setting:
    """Setting of the training."""

    configurations: list[str] = field(default_factory=lambda: ["training.cfg"])
    species: list[int] = field(default_factory=list)
    potential_initial: str = "initial.mtp"
    potential_final: str = "final.mtp"
    seed: int | None = None
    engine: str = "numpy"
    loss: dict[str, Any] = field(
        default_factory=lambda: {
            "energy-weight": 1.0,
            "force-weight": 0.01,
            "stress-weight": 0.001,
            "stress-times-volume": False,
        },
    )
    steps: list[dict] = field(
        default_factory=lambda: [
            {"method": "L-BFGS-B", "optimized": ["radial_coeffs", "moment_coeffs"]},
        ],
    )


def parse_setting(filename: str) -> Setting:
    """Parse setting file."""
    with pathlib.Path(filename).open("rb") as f:
        setting_overwritten = tomllib.load(f)

    if isinstance(setting_overwritten["configurations"], str):
        setting_overwritten["configurations"] = [setting_overwritten["configurations"]]

    # convert the old style "steps" like {'steps`: ['L-BFGS-B']} to the new one
    # {'steps`: {'method': 'L-BFGS-B'}
    # Default 'optimized' is defined in each `Optimizer` class.
    for i, value in enumerate(setting_overwritten["steps"]):
        if not isinstance(value, dict):
            setting_overwritten["steps"][i] = {"method": value}

    return Setting(**setting_overwritten)
