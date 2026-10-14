"""Multi-GPU prediction with persistent workers over existing model windows."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .cli_config import (
    auto_nonnegative_int,
    auto_positive_int,
    env_auto_nonnegative_int,
    env_auto_positive_int,
    env_bool,
    env_int,
    env_str,
    is_auto,
    str_to_bool,
)
from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_FLANK_SIZE,
    DEFAULT_GENE_FILTER_MODE,
    DEFAULT_HMM_DECODE_BATCH,
    DEFAULT_SEQUENCE_LENGTH,
)

WORK_DIR_MARKER = ".oriongeno-multi-work-dir"
WORK_DIR_MARKER_CONTENT = "oriongeno-multi-v1\n"

def _auto_device_list():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible.lower() not in {"none", "nodevfiles", "-1"}:
        devices = [item.strip() for item in visible.split(",") if item.strip()]
        if devices:
            return devices
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ["0"]
    devices = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return devices or ["0"]


def parse_device_list(value):
    if is_auto(value):
        return _auto_device_list()
    devices = [item.strip() for item in str(value or "").split(",") if item.strip()]
    if not devices:
        raise ValueError("--devices must contain at least one GPU id.")
    return devices


def validate_multi_args(args):
    """Validate multi-GPU arguments and return the requested devices."""
    try:
        devices = parse_device_list(args.devices)
    except ValueError as error:
        logging.error(str(error))
        sys.exit(1)

    if not args.genome:
        logging.error("--genome or ORIONGENO_GENOME is required.")
        sys.exit(1)
    if not args.output:
        logging.error("--output or ORIONGENO_OUT is required.")
        sys.exit(1)
    if not args.checkpoint:
        logging.error("--checkpoint or ORIONGENO_CHECKPOINT is required.")
        sys.exit(1)
    if not args.output_gene and not args.output_repeat:
        logging.error("At least one of --output-gene or --output-repeat must be True.")
        sys.exit(1)
    return devices


def _path_is_within(path, directory):
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _validated_work_dir(work_dir, output_path):
    raw_work_dir = Path(os.path.abspath(os.path.expanduser(os.fspath(work_dir))))
    if raw_work_dir.is_symlink():
        raise RuntimeError(f"Refusing to use a symlink as --work-dir: {raw_work_dir}")

    resolved_work_dir = raw_work_dir.resolve()
    resolved_output = Path(output_path).expanduser().resolve()
    repo_root = Path(__file__).resolve().parent.parent
    current_dir = Path.cwd().resolve()
    dangerous_roots = {
        Path(resolved_work_dir.anchor),
        Path.home().resolve(),
        repo_root,
        current_dir,
    }
    if resolved_work_dir in dangerous_roots:
        raise RuntimeError(f"Refusing to use protected directory as --work-dir: {resolved_work_dir}")
    if _path_is_within(repo_root, resolved_work_dir) or _path_is_within(
        current_dir,
        resolved_work_dir,
    ):
        raise RuntimeError(
            f"Refusing to use a parent of the project or current directory as --work-dir: "
            f"{resolved_work_dir}"
        )
    if _path_is_within(resolved_output, resolved_work_dir):
        raise RuntimeError(
            f"Final output must be outside --work-dir because successful runs remove it: "
            f"{resolved_output}"
        )
    return resolved_work_dir


def _work_dir_marker_path(work_dir):
    return Path(work_dir) / WORK_DIR_MARKER


def _has_valid_work_dir_marker(work_dir):
    marker_path = _work_dir_marker_path(work_dir)
    if not marker_path.is_file():
        return False
    try:
        return marker_path.read_text(encoding="utf-8") == WORK_DIR_MARKER_CONTENT
    except OSError:
        return False


def _mark_work_dir(work_dir):
    _work_dir_marker_path(work_dir).write_text(
        WORK_DIR_MARKER_CONTENT,
        encoding="utf-8",
    )


def prepare_work_dir(work_dir, output_path, keep_work_dir):
    """Create a dedicated work directory, safely replacing prior run data."""
    work_path = _validated_work_dir(work_dir, output_path)
    if work_path.exists():
        if not work_path.is_dir():
            raise RuntimeError(f"--work-dir exists but is not a directory: {work_path}")
        if not _has_valid_work_dir_marker(work_path) and any(work_path.iterdir()):
            unknown_entries = sorted(entry.name for entry in work_path.iterdir())
            raise RuntimeError(
                "Refusing to delete or reuse an unrecognized --work-dir. "
                f"Unknown entries: {unknown_entries[:5]}"
            )
        if not keep_work_dir:
            shutil.rmtree(work_path)

    work_path.mkdir(parents=True, exist_ok=True)
    _mark_work_dir(work_path)
    return os.fspath(work_path)


def cleanup_work_dir(work_dir, output_path):
    """Remove a work directory only when this tool has marked it as owned."""
    work_path = _validated_work_dir(work_dir, output_path)
    if not work_path.exists():
        return
    if not _has_valid_work_dir_marker(work_path):
        logging.warning(
            "Work directory marker is missing or invalid; refusing automatic cleanup: %s",
            work_path,
        )
        return
    shutil.rmtree(work_path)


def build_prediction_args(args):
    """Forward public options to the shared prediction and annotation runner."""
    command = [
        "--genome",
        os.path.abspath(args.genome),
        "--output",
        os.path.abspath(args.output),
        "--checkpoint",
        os.path.abspath(args.checkpoint),
        "--length",
        str(args.seq_len),
        "--flank",
        str(args.flank_size),
        "--batch-size",
        str(args.batch_size),
        "--output-gene",
        str(args.output_gene),
        "--output-repeat",
        str(args.output_repeat),
        "--gene-filter-mode",
        args.gene_filter_mode,
    ]
    if getattr(args, "hmm_parallel_factor", 0):
        command.extend(["--hmm-parallel-factor", str(args.hmm_parallel_factor)])
    if getattr(args, "hmm_decode_batch", None) is not None:
        command.extend(["--hmm-decode-batch", str(args.hmm_decode_batch)])
    if hasattr(args, "profile_hmm"):
        command.extend(["--profile-hmm", str(args.profile_hmm)])
    if getattr(args, "window_sizing", "legacy") != "legacy":
        command.extend(["--window-sizing", args.window_sizing])
    if args.species_name:
        command.extend(["--species-name", args.species_name])
    return command


def run_multi_prediction(args):
    from .window_pool import run_window_prediction

    return run_window_prediction(args, validate_multi_args(args))

def build_multi_parser(prog="main.py multi"):
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run OrionGeno on multiple GPUs with persistent workers.",
    )
    parser.add_argument("--window-task-size", type=int, default=32,
                        help="Maximum windows per worker job with multiple GPUs; "
                        "short waves are balanced across devices.")
    parser.add_argument("--window-sizing", choices=("legacy", "compact-short", "compact"),
                        default="legacy", help="Optional N-padding reduction; may change predictions.")
    parser.add_argument("--genome", default=env_str("ORIONGENO_GENOME", ""))
    parser.add_argument("--output", default=env_str("ORIONGENO_OUT", ""))
    parser.add_argument("--checkpoint", default=env_str("ORIONGENO_CHECKPOINT", ""))
    parser.add_argument(
        "--devices",
        default=env_str("ORIONGENO_DEVICES", "auto"),
        help="Comma-separated GPU IDs, or 'auto' to use CUDA_VISIBLE_DEVICES/nvidia-smi.",
    )
    parser.add_argument(
        "--length",
        dest="seq_len",
        type=int,
        default=env_int("ORIONGENO_SEQ_LEN", DEFAULT_SEQUENCE_LENGTH),
    )
    parser.add_argument(
        "--flank",
        dest="flank_size",
        type=int,
        default=env_int("ORIONGENO_FLANK_SIZE", DEFAULT_FLANK_SIZE),
    )
    parser.add_argument(
        "--batch-size",
        type=auto_positive_int,
        default=env_auto_positive_int("ORIONGENO_BATCH_SIZE", DEFAULT_BATCH_SIZE),
        help="Model-forward batch size per GPU. Use 'auto' to estimate once inside each persistent worker.",
    )
    parser.add_argument(
        "--hmm-parallel-factor",
        type=int,
        default=env_int("ORIONGENO_HMM_PARALLEL_FACTOR", 0),
        help="Override the HMM chunk-parallel factor. Use 0 to choose it automatically.",
    )
    parser.add_argument(
        "--hmm-decode-batch",
        type=auto_nonnegative_int,
        default=env_auto_nonnegative_int(
            "ORIONGENO_HMM_DECODE_BATCH",
            DEFAULT_HMM_DECODE_BATCH,
        ),
        help=(
            "HMM Viterbi decode batch size, decoupled from --batch-size. Larger "
            "values improve CPU Viterbi batch-level parallelism and amortize "
            "per-step CUDA kernel-launch overhead on the GPU fallback path; "
            "CUDA OOM is handled by halving and retrying. "
            "Use 'auto' for a conservative tuned value; use 0 to reuse the "
            "model-forward batch size (legacy behavior)."
        ),
    )
    parser.add_argument(
        "--profile-hmm",
        type=str_to_bool,
        default=env_bool("ORIONGENO_PROFILE_HMM", False),
        help="Print per-batch HMM timing breakdowns.",
    )
    parser.add_argument("--output-gene", type=str_to_bool, default=env_bool("ORIONGENO_OUTPUT_GENE", True))
    parser.add_argument("--output-repeat", type=str_to_bool, default=env_bool("ORIONGENO_OUTPUT_REPEAT", False))
    parser.add_argument(
        "--gene-filter-mode",
        choices=("strict", "none"),
        default=env_str(
            "ORIONGENO_GENE_FILTER_MODE",
            DEFAULT_GENE_FILTER_MODE,
        ).lower(),
        help=(
            "Gene annotation filtering mode. 'strict' writes only the strict-filtered "
            "records to --output; 'none' writes unfiltered predictions to --output."
        ),
    )
    parser.add_argument("--species-name", default=env_str("ORIONGENO_SPECIES_NAME", ""))
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--keep-work-dir", action="store_true")
    return parser
