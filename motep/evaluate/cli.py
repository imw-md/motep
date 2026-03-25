"""`motep evaluate`."""

import argparse

from motep.parallel import world
from motep.utils import measure_time

from .evaluator import evaluate_from_setting


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments."""
    parser.add_argument("setting")


def run(args: argparse.Namespace) -> None:
    """Run."""
    with measure_time("total"):
        evaluate_from_setting(args.setting, comm=world)


def main() -> None:
    """Command."""
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter_class)
    add_arguments(parser)
    args = parser.parse_args()
    run(args)
