"""`motep`."""

import argparse
import logging
import sys

import motep.evaluate.cli
import motep.grade.cli
import motep.train.cli
import motep.upconvert.cli

logger = logging.getLogger("motep")
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
        "train": motep.train.cli,
        "evaluate": motep.evaluate.cli,
        "grade": motep.grade.cli,
        "upconvert": motep.upconvert.cli,
    }
    for key, value in commands.items():
        value.add_arguments(subparsers.add_parser(key))

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    commands[args.command].run(args)
