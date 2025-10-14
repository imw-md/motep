"""Functions related to the setting file."""

from __future__ import annotations

import pathlib
import tomllib
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize._minimize import MINIMIZE_METHODS  # noqa: PLC2701


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
class UpconvertPotentials:
    """Setting of the potentials."""

    base: str = "base.mtp"
    initial: str = "initial.mtp"
    final: str = "final.mtp"


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
    rng: np.random.Generator | None = None
    engine: str = "numba"

    def __post_init__(self) -> None:
        """Postprocess attributes."""
        self.rng = self.rng or np.random.default_rng(self.seed)


def _convert_steps(steps: list[dict]) -> list[dict]:
    for i, value in enumerate(steps):
        if not isinstance(value, dict):
            steps["steps"][i] = {"method": value}
        if value["method"].lower() in MINIMIZE_METHODS:
            if "kwargs" not in value:
                value["kwargs"] = {}
            value["kwargs"]["method"] = value["method"]
            value["method"] = "minimize"
    return steps


@dataclass
class TrainSetting(Setting):
    """Setting of the training."""

    loss: LossSetting = field(default_factory=LossSetting)
    steps: list[dict] = field(
        default_factory=lambda: [
            {"method": "minimize"},
        ],
    )

    def __post_init__(self) -> None:
        """Postprocess attributes."""
        super().__post_init__()
        if isinstance(self.loss, dict):
            self.loss = LossSetting(**self.loss)

        # Default 'optimized' is defined in each `Optimizer` class.

        # convert the old style "steps" like {'steps`: ['L-BFGS-B']} to the new one
        # {'steps`: {'method': 'L-BFGS-B'}
        self.steps = _convert_steps(self.steps)


@dataclass
class ApplySetting(Setting):
    """Setting for the application of the potential."""


@dataclass
class GradeSetting(Setting):
    """Setting for the extrapolation-grade calculations."""

    algorithm: str = "maxvol"


@dataclass
class UpconvertSetting:
    """Setting for the upconversion."""

    potentials: UpconvertPotentials = field(default_factory=UpconvertPotentials)

    def __post_init__(self) -> None:
        """Postprocess attributes."""
        if isinstance(self.potentials, dict):
            self.potentials = UpconvertPotentials(**dict(self.potentials))


def parse_setting(filename: str) -> dict:
    """Parse setting file.

    Returns
    -------
    dict

    """
    with pathlib.Path(filename).open("rb") as f:
        setting_overwritten = tomllib.load(f)

    # convert the data files to lists
    keys = ["data_training", "data_in", "data_out"]
    for key in keys:
        if key in setting_overwritten and isinstance(setting_overwritten[key], str):
            setting_overwritten[key] = [setting_overwritten[key]]

    return setting_overwritten


def load_setting_train(filename: str) -> TrainSetting:
    """Load setting for `train`.

    Returns
    -------
    TrainSetting

    """
    return TrainSetting(**parse_setting(filename))


def load_setting_apply(filename: str) -> ApplySetting:
    """Load setting for `grade`.

    Returns
    -------
    GradeSetting

    """
    return ApplySetting(**parse_setting(filename))


def load_setting_grade(filename: str) -> GradeSetting:
    """Load setting for `grade`.

    Returns
    -------
    GradeSetting

    """
    return GradeSetting(**parse_setting(filename))


def load_setting_upconvert(filename: str | None) -> UpconvertSetting:
    """Load setting for `upconvert`.

    Returns
    -------
    UpconvertSetting

    """
    if filename is None:
        return UpconvertSetting()
    return UpconvertSetting(**parse_setting(filename))
