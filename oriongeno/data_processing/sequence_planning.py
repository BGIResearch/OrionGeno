"""Sequence-window planning helpers for OrionGeno prediction."""

from __future__ import annotations

import logging
import math
import sys

from ..constants import DEFAULT_PARALLEL_FACTOR, SEQUENCE_GROUP_SIZE


def compute_parallel_factor(model_seq_len, core_seq_len=None):
    """Return an HMM factor that keeps model windows on 51200-base multiples."""
    core_seq_len = int(core_seq_len if core_seq_len is not None else model_seq_len)
    model_seq_len = int(model_seq_len)
    preferred = min(DEFAULT_PARALLEL_FACTOR, model_seq_len)

    for factor in range(preferred, 0, -1):
        if model_seq_len % factor != 0:
            continue
        if core_seq_len % (512 * factor) != 0:
            continue
        return factor

    for factor in range(preferred, 0, -1):
        if model_seq_len % factor == 0:
            return factor
    return 1


def check_parallel_factor(parallel_factor, seq_len, core_seq_len=None):
    if parallel_factor < 2 or 2 * parallel_factor > seq_len:
        logging.info(
            "The parallel factor is unusual for this sequence length: %s",
            parallel_factor,
        )
    if seq_len % parallel_factor != 0:
        logging.error(
            "parallel-factor must divide seq-len. Got parallel-factor=%s, seq-len=%s.",
            parallel_factor,
            seq_len,
        )
        sys.exit(1)
    if parallel_factor < 1:
        logging.error("parallel-factor must be greater than zero.")
        sys.exit(1)
    if core_seq_len is not None and core_seq_len % (512 * parallel_factor) != 0:
        logging.error(
            "core seq-len must be divisible by 512 * parallel-factor. "
            "Got core-seq-len=%s, parallel-factor=%s.",
            core_seq_len,
            parallel_factor,
        )
        sys.exit(1)


def check_flank_size(flank_size, seq_len):
    if flank_size < 0:
        logging.error("flank-size must be non-negative.")
        sys.exit(1)
    if flank_size > 0 and flank_size >= seq_len:
        logging.info(
            "flank-size is large relative to seq-len: flank-size=%s, seq-len=%s.",
            flank_size,
            seq_len,
        )


def adapted_core_chunk_size(
    max_seq_len,
    chunk_size=512000,
    parallel_factor=1,
    *,
    upper_only=True,
    overlap=0,
):
    """Return the core chunk size that GenomeSequences will use for a group."""
    max_seq_len = int(max_seq_len)
    chunk_size = int(chunk_size)
    parallel_factor = max(1, int(parallel_factor or 1))
    overlap = max(0, int(overlap or 0))

    if max_seq_len >= chunk_size:
        return chunk_size

    adapted = max_seq_len
    if adapted <= 2 * overlap:
        adapted = min(2 * overlap + 1, chunk_size)
    if adapted < 2 * parallel_factor:
        adapted = 2 * parallel_factor

    if upper_only:
        divisor = 512 * parallel_factor
    else:
        divisor = 2 * 9 * parallel_factor // math.gcd(18, parallel_factor)
    return divisor * (1 + (adapted - 1) // divisor)


def _sequence_group_cost(seq_len, core_chunk_size, flank_size=0):
    window_count = max(1, int(math.ceil(int(seq_len) / int(core_chunk_size))))
    model_chunk_size = int(core_chunk_size) + 2 * max(0, int(flank_size or 0))
    return window_count * model_chunk_size


def group_sequences(
    seq_names,
    seq_lens,
    target_size=SEQUENCE_GROUP_SIZE,
    chunk_size=512000,
    parallel_factor=1,
    *,
    upper_only=True,
    flank_size=0,
):
    """Group sequences by the model window size they will actually use.

    Short scaffolds are bucketed by their adaptive chunk size. This avoids
    treating every short sequence as if it consumed a full ``chunk_size`` window
    and lets many small scaffolds share a group without increasing padding.
    """
    target_size = max(1, int(target_size))
    sorted_seqs = sorted(
        (
            (
                adapted_core_chunk_size(
                    seq_len,
                    chunk_size=chunk_size,
                    parallel_factor=parallel_factor,
                    upper_only=upper_only,
                ),
                int(seq_len),
                seq_name,
            )
            for seq_name, seq_len in zip(seq_names, seq_lens)
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )
    groups = []
    current_group = []
    current_core_chunk_size = None
    current_sum = 0

    for core_chunk_size, seq_len, seq_name in sorted_seqs:
        cost = _sequence_group_cost(seq_len, core_chunk_size, flank_size=flank_size)
        should_flush = (
            current_group
            and (
                core_chunk_size != current_core_chunk_size
                or current_sum + cost > target_size
            )
        )
        if should_flush:
            groups.append(current_group)
            current_group = []
            current_core_chunk_size = None
            current_sum = 0

        current_group.append(seq_name)
        current_core_chunk_size = core_chunk_size
        current_sum += cost

    if current_group:
        groups.append(current_group)
    return groups
