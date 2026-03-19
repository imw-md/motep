"""`motep grade` command."""

import argparse

from motep.grader.grader import grade


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments."""
    parser.add_argument("setting")


def run(args: argparse.Namespace) -> None:
    """Run."""
    grade(args.setting)
