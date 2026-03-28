from dataclasses import dataclass, field
from pathlib import Path

from scipy.optimize._minimize import MINIMIZE_METHODS  # noqa: PLC2701

from motep.loss import LossSetting
from motep.setting import (
    CommonSetting,
    ConfigurationsBase,
    DataclassFromAny,
    parse_setting,
)


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
class _Configurations(ConfigurationsBase):
    """Configurations."""

    training: list[str] = field(default_factory=lambda: ["training.cfg"])


@dataclass
class _Potentials(DataclassFromAny):
    """Potentials."""

    initial: str = "initial.mtp"
    final: str = "final.mtp"


@dataclass
class _Setting(DataclassFromAny):
    """Setting of the training."""

    common: CommonSetting = field(default_factory=CommonSetting)
    configurations: _Configurations = field(default_factory=_Configurations)
    potentials: _Potentials = field(default_factory=_Potentials)
    loss: LossSetting = field(default_factory=LossSetting)
    steps: list[dict] = field(default_factory=lambda: [{"method": "minimize"}])
    update_mindist: bool = False

    def __post_init__(self) -> None:
        """Postprocess attributes."""
        self.configurations = _Configurations.from_any(self.configurations)
        self.potentials = _Potentials.from_any(self.potentials)
        self.loss = LossSetting.from_any(self.loss)

        # Default 'optimized' is defined in each `Optimizer` class.

        # convert the old style "steps" like {'steps`: ['L-BFGS-B']} to the new one
        # {'steps`: {'method': 'L-BFGS-B'}
        self.steps = _convert_steps(self.steps)


def load_setting_train(filename: str | Path | None = None) -> _Setting:
    """Load setting for `train`.

    Returns
    -------
    TrainSetting

    """
    if filename is None:
        return _Setting()
    return _Setting(**parse_setting(filename))
