"""OrionGeno command-line entry point."""

from __future__ import annotations

import logging
import os
import sys

from . import __version__, prediction_runner

_PREDICTION_COMMANDS = {"predict", "prediction"}
_DISTRIBUTED_OPTIONS = {
    "--num-nodes", "--node-rank", "--distributed-timeout", "--distributed-poll-interval",
}


def _uses_distributed_options(args):
    return any(arg.split("=", 1)[0] in _DISTRIBUTED_OPTIONS for arg in args) or any(
        os.environ.get(name) for name in ("ORIONGENO_NUM_NODES", "ORIONGENO_NODE_RANK")
    )


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else list(argv)
    if args in (["--version"], ["-V"]):
        print(__version__)
        return

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    if args and args[0] in _PREDICTION_COMMANDS:
        args = args[1:]
    elif args and args[0] in {"multi", "multi-node"}:
        distributed = args[0] == "multi-node" or _uses_distributed_options(args[1:])
        if distributed:
            from .distributed_predict import build_multi_parser, run_multi_prediction
        else:
            from .multi_predict import build_multi_parser, run_multi_prediction
        parser = build_multi_parser()
        run_multi_prediction(parser.parse_args(args[1:]))
        return
    elif args and not args[0].startswith("-"):
        raise SystemExit(
            f"Unknown OrionGeno command: {args[0]}. "
            "Known commands: predict, prediction, multi, multi-node."
        )

    parser = prediction_runner.build_predict_parser()
    prediction_runner.run_prediction(parser.parse_args(args))


if __name__ == "__main__":
    main()
