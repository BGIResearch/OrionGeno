#!/usr/bin/env python3
"""OrionGeno command dispatcher."""

from __future__ import annotations

import logging
import sys

from . import merge_cli, multi_runner, prediction_runner, route_cli


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

INTERNAL_ARG_FLAGS = {
    "--checkpoint",
    "--seq-len",
    "--flank-size",
    "--parallel-factor",
    "--strand",
    "--use-hmm",
    "--upper-only",
    "--use-species-embedding",
    "--clamsa",
    "--coding-seq",
    "--protein-seq",
    "--id-prefix",
    "--min-seq-len",
    "--num-shards",
    "--shard-index",
    "--shard-strategy",
    "--shard-manifest",
    "--show-model-summary",
}


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else list(argv)
    if args and args[0] == "pipeline":
        parser = prediction_runner.build_external_parser(prog="oriongeno pipeline")
        external_args = parser.parse_args(args[1:])
        prediction_runner.run_prediction(
            prediction_runner.build_external_prediction_args(external_args)
        )
        return
    if args and args[0] in ("multi", "multigpu"):
        parser = multi_runner.build_multi_parser(prog=f"oriongeno {args[0]}")
        multi_runner.run_multi(parser.parse_args(args[1:]))
        return
    if args and args[0] == "route":
        route_cli.run_route(route_cli.build_route_parser().parse_args(args[1:]))
        return
    if args and args[0] == "merge":
        merge_cli.run_merge(merge_cli.build_merge_parser().parse_args(args[1:]))
        return
    if args and args[0] in ("predict", "internal"):
        parser = prediction_runner.build_predict_parser(prog=f"oriongeno {args[0]}")
        prediction_runner.run_prediction(parser.parse_args(args[1:]))
        return
    if any(arg.split("=", 1)[0] in INTERNAL_ARG_FLAGS for arg in args):
        parser = prediction_runner.build_predict_parser()
        prediction_runner.run_prediction(parser.parse_args(args))
        return

    external_args = prediction_runner.build_external_parser().parse_args(args)
    prediction_runner.run_prediction(
        prediction_runner.build_external_prediction_args(external_args)
    )


if __name__ == "__main__":
    main()
