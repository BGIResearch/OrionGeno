#!/usr/bin/env python3
"""OrionGeno command-line entry point."""

from __future__ import annotations

import logging
import sys

from . import prediction_runner


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

_PREDICTION_COMMANDS = {"predict", "prediction"}
_COMMAND_ALIASES = {"multi"}


def _unknown_command_error(command: str) -> SystemExit:
    commands = ", ".join(sorted(_COMMAND_ALIASES | _PREDICTION_COMMANDS))
    return SystemExit(
        f"Unknown OrionGeno command: {command}. "
        f"Known commands: {commands}. "
        "Use no command or 'predict' for single-genome prediction."
    )


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else list(argv)
    if args and args[0] in _PREDICTION_COMMANDS:
        parser = prediction_runner.build_predict_parser()
        prediction_runner.run_prediction(parser.parse_args(args[1:]))
        return
    if args and args[0] == "multi":
        from .multi_predict import build_multi_parser, run_multi_prediction

        parser = build_multi_parser()
        run_multi_prediction(parser.parse_args(args[1:]))
        return
    if args and not args[0].startswith("-"):
        raise _unknown_command_error(args[0])

    parser = prediction_runner.build_predict_parser()
    prediction_runner.run_prediction(parser.parse_args(args))


if __name__ == "__main__":
    main()
