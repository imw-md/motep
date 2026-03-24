"""`motep train`."""

import argparse

from motep.train.trainer import train
from motep.utils import measure_time


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments."""
    parser.add_argument("setting")


def run(args: argparse.Namespace) -> None:
    """Run."""
    with measure_time("total"):
        train(args.setting)


def main() -> None:
    """Command."""
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter_class)
    add_arguments(parser)
    args = parser.parse_args()
    run(args)
