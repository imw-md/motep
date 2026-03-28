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

    species: list[int] = field(default_factory=list)
    seed: int | None = None
    engine: str = "cext"


def parse_setting(filename: str | Path) -> dict[str, Any]:
    """Parse setting file.

    Returns
    -------
    dict

    """
    with Path(filename).open("rb") as f:
        return tomllib.load(f)
