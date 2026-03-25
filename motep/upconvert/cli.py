"""`motep upconvert`."""

import argparse

from motep.parallel import world
from motep.upconvert.upconverter import upconvert_from_setting
from motep.utils import measure_time


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments."""
    parser.add_argument("setting", nargs="?")


def run(args: argparse.Namespace) -> None:
    """Run."""
    with measure_time("total"):
        upconvert_from_setting(args.setting, comm=world)


def main() -> None:
    """Command."""
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter_class)
    add_arguments(parser)
    args = parser.parse_args()
    run(args)
