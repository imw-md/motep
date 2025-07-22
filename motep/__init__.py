"""motep."""

import argparse

from motep import (
    applier,
    grader,
    trainer,
    upconverter,
)


def main() -> None:
    """Command."""
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter_class)
    subparsers = parser.add_subparsers(dest="command")

    commands = {
        "train": trainer,
        "apply": applier,
        "grade": grader,
        "upconvert": upconverter,
    }
    for key, value in commands.items():
        value.add_arguments(subparsers.add_parser(key))

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    commands[args.command].run(args)
