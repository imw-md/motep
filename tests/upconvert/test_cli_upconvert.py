"""Tests for `motep upconvert`."""

from pathlib import Path

from motep.upconvert.upconverter import load_setting_upconvert


def test_example_upconvert(doc_path: Path) -> None:
    """Test if the input file offered in the documentation is parsable."""
    path = doc_path / "cli/motep.upconvert.toml"
    setting = load_setting_upconvert(path)
    assert not isinstance(setting.potentials, dict)  # converted to a dataclass?
    assert isinstance(setting.potentials.final, str)
