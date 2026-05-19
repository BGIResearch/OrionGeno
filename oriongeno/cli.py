#!/usr/bin/env python3
"""OrionGeno command-line entry point."""

from __future__ import annotations

import logging
import sys

from . import prediction_runner


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else list(argv)
    if args and args[0] == "multi":
        from .multi_predict import build_multi_parser, run_multi_prediction

        parser = build_multi_parser()
        run_multi_prediction(parser.parse_args(args[1:]))
        return

    parser = prediction_runner.build_predict_parser()
    prediction_runner.run_prediction(parser.parse_args(args))


if __name__ == "__main__":
    main()
