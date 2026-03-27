"""Tests for setting.py."""

import pytest

from motep.loss import LossSetting
from motep.train.trainer import TrainSetting


@pytest.mark.parametrize("kwargs", [{}, {"loss": {}}, {"loss": LossSetting()}])
def test_loss_setting(kwargs: dict) -> None:
    """Test if `LossSetting` is correctly parsed."""
    setting = TrainSetting(**kwargs)
    assert setting.loss == LossSetting()
