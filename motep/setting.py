"""Functions related to the setting file."""

import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
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
class CommonSetting(DataclassFromAny):
    """Setting of the training."""

    species: list[int] = field(default_factory=list)
    seed: int | None = None
    engine: str = "cext"


@dataclass
class ConfigurationsBase(DataclassFromAny):
    """Base class of the setting for the configurations."""

    def __post_init__(self) -> None:
        """Postprocess attributes."""
        # convert the data files to lists
        for key, value in asdict(self).items():
            if isinstance(value, str):
                setattr(self, key, [value])


def parse_setting(filename: str | Path) -> dict[str, Any]:
    """Parse setting file.

    Returns
    -------
    dict

    """
    with Path(filename).open("rb") as f:
        return tomllib.load(f)
