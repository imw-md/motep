"""Tests for setting.py."""

import pytest

from motep.setting import LossSetting, TrainSetting


@pytest.mark.parametrize("kwargs", [{}, {"loss": {}}, {"loss": LossSetting()}])
def test_loss_setting(kwargs: dict) -> None:
    """Test if `LossSetting` is correctly parsed."""
    setting = TrainSetting(**kwargs)
    assert setting.loss == LossSetting()


@pytest.mark.parametrize(
    "steps",
    [
        [{"method": "minimize"}],
        [{"method": "l-bfgs-b"}],
        [{"method": "minimize", "kwargs": {"method": "l-bfgs-b"}}],
    ],
)
def test_train_steps_setting(steps: list) -> None:
    """Test if `steps` in `TrainSetting` is correctly parsed."""
    setting = TrainSetting(steps=steps)
    assert setting.steps[0]["method"] == "minimize"
