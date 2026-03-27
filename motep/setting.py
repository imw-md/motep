"""Functions related to the setting file."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self


class DataclassFromAny:
    """Mixin to create class from any."""

    @classmethod
    def from_any(
        cls: type[Self],
        value: Self | Mapping[str, Any] | None = None,
    ) -> Self:
        """Create instance from `value`.

        Returns
        -------
        Self

        """
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(**value)
        return cls()


@dataclass
class Setting(DataclassFromAny):
    """Setting of the training."""

    data_training: list[str] = field(default_factory=lambda: ["training.cfg"])
    data_in: list[str] = field(default_factory=lambda: ["in.cfg"])
    data_out: list[str] = field(default_factory=lambda: ["out.cfg"])
    species: list[int] = field(default_factory=list)
    potential_initial: str = "initial.mtp"
    potential_final: str = "final.mtp"
    seed: int | None = None
    engine: str = "cext"


def parse_setting(filename: str | Path) -> dict[str, Any]:
    """Parse setting file.

    Returns
    -------
    dict

    """
    with Path(filename).open("rb") as f:
        setting_overwritten = tomllib.load(f)

    # convert the data files to lists
    keys = ["data_training", "data_in", "data_out"]
    for key in keys:
        if key in setting_overwritten and isinstance(setting_overwritten[key], str):
            setting_overwritten[key] = [setting_overwritten[key]]

    return setting_overwritten
