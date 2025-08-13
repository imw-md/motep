"""Tests for setting.py."""

import pytest

from motep.setting import LossSetting, TrainSetting


@pytest.mark.parametrize("kwargs", [{}, {"loss": {}}, {"loss": LossSetting()}])
def test_loss_setting(kwargs: dict) -> None:
    """Test if `LossSetting` is correctly parsed."""
    setting = TrainSetting(**kwargs)
    assert setting.loss == LossSetting()
