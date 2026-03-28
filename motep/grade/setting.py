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
class _Configurations(DataclassFromAny):
    """Configurations."""

    training: list[str] = field(default_factory=lambda: ["training.cfg"])
    initial: list[str] = field(default_factory=lambda: ["initial.cfg"])
    final: list[str] = field(default_factory=lambda: ["final.cfg"])


@dataclass
class _Potentials(DataclassFromAny):
    """Setting of the potentials."""

    final: str = "final.mtp"


@dataclass
class _GradeSetting(DataclassFromAny):
    """Setting for the extrapolation-grade calculations."""

    mode: GradeMode = GradeMode.CONFIGURATION
    maxvol: MaxVolSetting = field(default_factory=MaxVolSetting)

    def __post_init__(self) -> None:
        """Postprocess attributes."""
        self.maxvol = MaxVolSetting.from_any(self.maxvol)


@dataclass
class _Setting(DataclassFromAny):
    """Setting for the extrapolation-grade calculations."""

    common: CommonSetting = field(default_factory=CommonSetting)
    configurations: _Configurations = field(default_factory=_Configurations)
    potentials: _Potentials = field(default_factory=_Potentials)
    grade: _GradeSetting = field(default_factory=_GradeSetting)

    def __post_init__(self) -> None:
        """Postprocess attributes."""
        self.configurations = _Configurations.from_any(self.configurations)
        self.potentials = _Potentials.from_any(self.potentials)


def load_setting_grade(filename: str | Path | None = None) -> _Setting:
    """Load setting for `grade`.

    Returns
    -------
    GradeSetting

    """
    if filename is None:
        return _Setting()
    return _Setting(**parse_setting(filename))
