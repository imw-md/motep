from dataclasses import dataclass, field
from pathlib import Path

from motep.setting import DataclassFromAny, parse_setting


@dataclass
class UpconvertPotentials(DataclassFromAny):
    """Setting of the potentials."""

    base: str = "base.mtp"
    initial: str = "initial.mtp"
    final: str = "final.mtp"


@dataclass
class Setting(DataclassFromAny):
    """Setting for the upconversion."""

    potentials: UpconvertPotentials = field(default_factory=UpconvertPotentials)

    def __post_init__(self) -> None:
        """Postprocess attributes."""
        self.potentials = UpconvertPotentials.from_any(self.potentials)


def load_setting_upconvert(filename: str | Path | None = None) -> Setting:
    """Load setting for `upconvert`.

    Returns
    -------
    UpconvertSetting

    """
    if filename is None:
        return Setting()
    return Setting(**parse_setting(filename))
