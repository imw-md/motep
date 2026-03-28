from dataclasses import dataclass, field
from pathlib import Path

from motep.setting import DataclassFromAny, parse_setting


@dataclass
class _Potentials(DataclassFromAny):
    """Setting of the potentials."""

    training: str = "training.mtp"
    initial: str = "initial.mtp"
    final: str = "final.mtp"


@dataclass
class _Setting(DataclassFromAny):
    """Setting for the upconversion."""

    potentials: _Potentials = field(default_factory=_Potentials)

    def __post_init__(self) -> None:
        """Postprocess attributes."""
        self.potentials = _Potentials.from_any(self.potentials)


def load_setting_upconvert(filename: str | Path | None = None) -> _Setting:
    """Load setting for `upconvert`.

    Returns
    -------
    UpconvertSetting

    """
    if filename is None:
        return _Setting()
    return _Setting(**parse_setting(filename))
