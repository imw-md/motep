"""`motep`."""

import argparse
import logging
import sys

import motep.grade.cli as grade_cli
import motep.train.cli as train_cli
from motep import applier, evaluator, upconverter

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
logger.addHandler(stdout_handler)


def main() -> None:
    """Command."""
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter_class)
    subparsers = parser.add_subparsers(dest="command")

    commands = {
        "train": train_cli,
        "apply": applier,
        "evaluate": evaluator,
        "grade": grade_cli,
        "upconvert": upconverter,
    }
    for key, value in commands.items():
        value.add_arguments(subparsers.add_parser(key))

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    commands[args.command].run(args)
