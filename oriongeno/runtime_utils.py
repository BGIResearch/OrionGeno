"""Distributed runtime helpers for OrienGeno inference.

Authors: wangshengfu, caixudong
"""

import glob
import logging
import os
import pickle
import shutil
import time
from dataclasses import dataclass

import torch.distributed as dist


@dataclass
class RuntimeContext:
    """Runtime metadata shared across single-process and distributed inference."""

    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: str
    backend: str = ""
    work_dir: str = ""

    @property
    def is_main(self):
        return self.rank == 0

    @property
    def log_prefix(self):
        return f"[rank {self.rank}]"


def shard_work(items, rank, world_size):
    """Split a task list across ranks using round-robin sharding."""
    if world_size <= 1:
        return items
    return items[rank::world_size]


def dist_barrier(runtime):
    """Synchronize all ranks when distributed inference is active."""
    if runtime.enabled and dist.is_initialized():
        dist.barrier()


def cleanup_runtime(runtime):
    """Release distributed resources once inference is complete."""
    if runtime.enabled and dist.is_initialized():
        dist_barrier(runtime)
        if runtime.is_main and runtime.work_dir:
            try:
                shutil.rmtree(runtime.work_dir)
            except FileNotFoundError:
                pass
            except OSError:
                pass
        dist_barrier(runtime)
        dist.destroy_process_group()


def configure_rank_logging(runtime):
    """Silence info-level logs on non-main distributed ranks."""
    if runtime.enabled and not runtime.is_main:
        logging.getLogger().setLevel(logging.WARNING)


def emit_rank_progress(runtime, message):
    """Print one concise progress line from any rank."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - {runtime.log_prefix} {message}", flush=True)


def format_sequence_group_label(sequence_names, max_names=3):
    """Summarize a sequence group for concise progress logging."""
    if not sequence_names:
        return "<none>"
    if len(sequence_names) <= max_names:
        return ", ".join(sequence_names)
    shown = ", ".join(sequence_names[:max_names])
    remaining = len(sequence_names) - max_names
    return f"{shown} (+{remaining} more)"


def rank_payload_path(runtime, stage_name):
    """Return the per-rank pickle path used for result exchange."""
    return os.path.join(runtime.work_dir, f"{stage_name}.rank{runtime.rank}.pkl")


def prepare_dist_work_dir(runtime):
    """Create a clean shared workspace for per-rank payloads."""
    if not runtime.enabled:
        return
    if runtime.is_main:
        os.makedirs(runtime.work_dir, exist_ok=True)
        for stale_path in glob.glob(os.path.join(runtime.work_dir, "*.rank*.pkl")):
            os.remove(stale_path)
    dist_barrier(runtime)


def save_rank_payload(runtime, stage_name, payload):
    """Persist one rank's intermediate results for rank0 aggregation."""
    if not runtime.enabled:
        return
    with open(rank_payload_path(runtime, stage_name), "wb") as handle:
        pickle.dump(payload, handle)


def load_all_rank_payloads(runtime, stage_name):
    """Load per-rank payloads in rank order from the shared work directory."""
    payloads = []
    for rank in range(runtime.world_size):
        payload_path = os.path.join(runtime.work_dir, f"{stage_name}.rank{rank}.pkl")
        if not os.path.exists(payload_path):
            raise FileNotFoundError(
                f"Missing distributed payload for stage '{stage_name}': {payload_path}"
            )
        with open(payload_path, "rb") as handle:
            payloads.append((rank, pickle.load(handle)))
    return payloads
