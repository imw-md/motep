from dataclasses import dataclass, field
from pathlib import Path

from motep.setting import CommonSetting, DataclassFromAny, parse_setting


@dataclass
class EvalConfigurations(DataclassFromAny):
    """Configurations."""

    initial: list[str] = field(default_factory=lambda: ["initial.cfg"])
    final: list[str] = field(default_factory=lambda: ["final.cfg"])


@dataclass
class EvalPotentials(DataclassFromAny):
    """Potentials."""

    final: str = "final.mtp"


@dataclass
class Setting(DataclassFromAny):
    """Setting for the application of the potential."""

    common: CommonSetting = field(default_factory=CommonSetting)
    configurations: EvalConfigurations = field(default_factory=EvalConfigurations)
    potentials: EvalPotentials = field(default_factory=EvalPotentials)

    def __post_init__(self) -> None:
        """Postprocess attributes."""
        self.configurations = EvalConfigurations.from_any(self.configurations)
        self.potentials = EvalPotentials.from_any(self.potentials)


def load_setting_evaluate(filename: str | Path | None = None) -> Setting:
    """Load setting for `evaluate`.

    Returns
    -------
    EvaluateSetting

    """
    if filename is None:
        return Setting()
    return Setting(**parse_setting(filename))
