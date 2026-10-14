"""Optional compute-only padding reduction, preserving genomic window starts."""
from collections import defaultdict
import json
import logging
import math
from pathlib import Path

import numpy as np

from .inference_runtime import predict_with_memory_backoff


def model_input_alignment(checkpoint):
    """The encoder pools once per scale; decoder doubles at every stage."""
    with (Path(checkpoint) / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    settings = config.get("settings", {})
    if settings.get("pooling_kernel_size", 2) != 2:
        raise ValueError("Compact windows require the supported factor-two pyramid")
    scales = settings.get("feature_scales", [1, 2, 4, 8, 16, 32, 64])
    if not scales:
        raise ValueError("The model must define at least one pyramid scale")
    return 2 ** len(scales)


def compact_model_length(valid_core, original_model, flank, alignment, sizing, core_start):
    if sizing == "legacy" or (sizing == "compact-short" and core_start != 1):
        return original_model
    # Coarse buckets amortize model batching across similarly sized scaffolds.
    quantum = math.lcm(4096, max(1, alignment))
    needed = valid_core + 2 * flank
    return min(original_model, ((needed + quantum - 1) // quantum) * quantum)


def crop_oriented_window(window, length, strand):
    # Tail padding moves to the left after reverse complementation.
    return window[:length] if strand == "+" else window[len(window) - length:]


def compact_prediction_groups(predictor, strand_inputs, context, max_chunks=0):
    buckets = defaultdict(list)
    original_bases = processed_bases = 0
    for item in strand_inputs:
        original_model = item["x_data"].shape[1]
        item["prediction_lengths"] = []
        for index, coord in enumerate(item["coords"]):
            valid = int(coord[3]) - int(coord[2]) + 1
            length = compact_model_length(valid, original_model, context.flank_size,
                                          context.window_alignment, context.window_sizing,
                                          int(coord[2]))
            item["prediction_lengths"].append(length)
            buckets[length].append((item, index))
            original_bases += original_model
            processed_bases += length
    for length, jobs in sorted(buckets.items(), reverse=True):
        predictor.adapt_batch_size(length)
        span = max_chunks or len(jobs)
        for left in range(0, len(jobs), span):
            selected = jobs[left:left + span]
            chunks = np.stack([crop_oriented_window(item["x_data"][index], length, item["strand"])
                               for item, index in selected])
            outputs = predict_with_memory_backoff(predictor, chunks, save=True,
                                                 output_gene=context.output_gene,
                                                 output_repeat=context.output_repeat)
            for row, (item, index) in enumerate(selected):
                destination = item.setdefault("prediction_outputs", {})
                for head, values in outputs.items():
                    if hasattr(values, "detach"):
                        values = values.detach().cpu().numpy()
                    if head not in destination:
                        destination[head] = np.zeros(item["x_data"].shape[:2], dtype=values.dtype)
                    start = 0 if item["strand"] == "+" else item["x_data"].shape[1] - length
                    destination[head][index, start:start + length] = values[row]
    logging.info("Window sizing %s: model bases %d -> %d (%.2f%% fewer)",
                 context.window_sizing, original_bases, processed_bases,
                 100 * (1 - processed_bases / max(1, original_bases)))


def compact_core_rows(rows, item, flank):
    """Trim the restored coordinate canvas to each actual predicted core."""
    lengths = item.get("prediction_lengths")
    if lengths is None:
        return rows
    return [crop_oriented_window(row, length - 2 * flank, item["strand"])
            for row, length in zip(rows, lengths)]
