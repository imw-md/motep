"""Functions related to the setting file."""

from __future__ import annotations

import pathlib
import tomllib
from dataclasses import dataclass, field
from inspect import signature
from typing import Any

scipy_minimize_methods = {
    "nelder-mead",
    "powell",
    "cg",
    "bfgs",
    "newton-cg",
    "l-bfgs-b",
    "tnc",
    "cobyla",
    "cobyqa",
    "slsqp",
    "trust-constr",
    "dogleg",
    "trust-ncg",
    "trust-exact",
    "trust-krylov",
}


@dataclass
class LossSetting:
    """Setting of the loss function."""

    energy_weight: float = 1.0
    forces_weight: float = 0.01
    stress_weight: float = 0.001
    energy_per_atom: bool = True
    forces_per_atom: bool = True
    stress_times_volume: bool = False
    energy_per_conf: bool = True
    forces_per_conf: bool = True
    stress_per_conf: bool = True


@dataclass
class Setting:
    """Setting of the training."""

    data_training: list[str] = field(default_factory=lambda: ["training.cfg"])
    data_in: list[str] = field(default_factory=lambda: ["in.cfg"])
    data_out: list[str] = field(default_factory=lambda: ["out.cfg"])
    species: list[int] = field(default_factory=list)
    potential_initial: str = "initial.mtp"
    potential_final: str = "final.mtp"
    seed: int | None = None
    engine: str = "numpy"


@dataclass
class TrainSetting(Setting):
    """Setting of the training."""

    loss: dict[str, Any] = field(default_factory=LossSetting)
    steps: list[dict] = field(
        default_factory=lambda: [
            {"method": "L-BFGS-B", "optimized": ["radial_coeffs", "moment_coeffs"]},
        ],
    )

    def __post_init__(self) -> None:
        """Postprocess attributes."""
        self.loss = LossSetting(**self.loss)


@dataclass
class GradeSetting(Setting):
    """Setting for the extrapolation-grade calculations."""

    algorithm: str = "maxvol"


def _parse_steps(setting_overwritten: dict) -> dict:
    for i, value in enumerate(setting_overwritten["steps"]):
        if not isinstance(value, dict):
            setting_overwritten["steps"][i] = {"method": value}
        if value["method"].lower() in scipy_minimize_methods:
            if "kwargs" not in value:
                value["kwargs"] = {}
            value["kwargs"]["method"] = value["method"]
            value["method"] = "minimize"
    return setting_overwritten


def parse_setting(filename: str) -> Setting:
    """Parse setting file."""
    with pathlib.Path(filename).open("rb") as f:
        setting_overwritten = tomllib.load(f)

    keys = ["data_training", "data_in", "data_out"]
    for key in keys:
        if key in setting_overwritten and isinstance(setting_overwritten[key], str):
            setting_overwritten[key] = [setting_overwritten[key]]

    # convert the old style "steps" like {'steps`: ['L-BFGS-B']} to the new one
    # {'steps`: {'method': 'L-BFGS-B'}
    # Default 'optimized' is defined in each `Optimizer` class.
    if "steps" in setting_overwritten:
        setting_overwritten = _parse_steps(setting_overwritten)
    return setting_overwritten


def load_setting_train(filename: str) -> TrainSetting:
    """Load setting for `train`."""
    return TrainSetting(**parse_setting(filename))


def load_setting_grade(filename: str) -> GradeSetting:
    """Load setting for `grade`."""
    return GradeSetting(**parse_setting(filename))
