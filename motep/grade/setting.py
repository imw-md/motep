from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from motep.setting import CommonSetting, DataclassFromAny, parse_setting

from .maxvol import MaxVolSetting


class GradeMode(StrEnum):
    """Extrapolation grade mode."""

    CONFIGURATION = "configuration"
    NEIGHBORHOOD = "neighborhood"


@dataclass
class GradeConfigurations(DataclassFromAny):
    """Configurations."""

    training: list[str] = field(default_factory=lambda: ["training.cfg"])
    initial: list[str] = field(default_factory=lambda: ["initial.cfg"])
    final: list[str] = field(default_factory=lambda: ["final.cfg"])


@dataclass
class GradePotentials(DataclassFromAny):
    """Setting of the potentials."""

    final: str = "final.mtp"


@dataclass
class GradeSetting(DataclassFromAny):
    """Setting for the extrapolation-grade calculations."""

    mode: GradeMode = GradeMode.CONFIGURATION
    maxvol: MaxVolSetting = field(default_factory=MaxVolSetting)

    def __post_init__(self) -> None:
        """Postprocess attributes."""
        self.maxvol = MaxVolSetting.from_any(self.maxvol)


@dataclass
class Setting(DataclassFromAny):
    """Setting for the extrapolation-grade calculations."""

    common: CommonSetting = field(default_factory=CommonSetting)
    configurations: GradeConfigurations = field(default_factory=GradeConfigurations)
    potentials: GradePotentials = field(default_factory=GradePotentials)
    grade: GradeSetting = field(default_factory=GradeSetting)

    def __post_init__(self) -> None:
        """Postprocess attributes."""
        self.configurations = GradeConfigurations.from_any(self.configurations)
        self.potentials = GradePotentials.from_any(self.potentials)


def load_setting_grade(filename: str | Path | None = None) -> Setting:
    """Load setting for `grade`.

    Returns
    -------
    GradeSetting

    """
    if filename is None:
        return Setting()
    return Setting(**parse_setting(filename))
