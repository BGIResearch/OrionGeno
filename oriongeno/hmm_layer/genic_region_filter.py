"""Genic region detection for targeted Viterbi decoding.

Implements windowed gene-state screening with interval expansion and merging.
Scans per-nucleotide gene probabilities to find candidate genic regions,
reducing Viterbi workload by skipping intergenic deserts.
"""

import numpy as np
import torch


def detect_genic_regions(
    gene_probabilities,
    min_avg_threshold=0.1,
    max_threshold=0.5,
    min_cds_length=50,
    step=50,
):
    """Detect ranges of potential genes from per-nucleotide probabilities.

    Args:
        gene_probabilities: [batch, seq_len, num_states] tensor or array.
            State 0 is intergenic (IR), states 1+ are genic.
        min_avg_threshold: Minimum average genic probability in a window.
        max_threshold: Threshold for counting high-confidence genic bases.
        min_cds_length: Minimum number of high-confidence bases to keep a region.
        step: Scanning window step size (bp).

    Returns:
        List of (start, end) tuples for each detected genic region (inclusive end).
        Returns empty list if no regions found.
    """
    # Convert to numpy if needed
    if torch.is_tensor(gene_probabilities):
        gene_probabilities = gene_probabilities.detach().cpu().numpy()

    # Handle batch dimension
    if gene_probabilities.ndim == 3:
        if gene_probabilities.shape[0] != 1:
            raise ValueError(
                f"detect_genic_regions expects batch=1, got {gene_probabilities.shape[0]}"
            )
        gene_probabilities = gene_probabilities[0]

    seq_length = gene_probabilities.shape[0]
    # Sum over genic states (states 1+, excluding state 0 = intergenic)
    genic_proba = gene_probabilities[:, 1:].astype(np.float32).sum(
        axis=1, dtype=np.float32
    )

    genic_regions = []
    genic_region_start = 0
    in_genic_region = False
    cumulative_sum = np.cumsum(genic_proba, dtype=np.float32)

    for start in range(0, seq_length, step):
        end = min(start + step, seq_length)
        if start == 0:
            window_mean = cumulative_sum[end - 1] / end
        else:
            window_mean = (cumulative_sum[end - 1] - cumulative_sum[start - 1]) / (
                end - start
            )

        if window_mean >= min_avg_threshold:
            in_genic_region = True
        else:
            if in_genic_region:
                potential_range = genic_proba[genic_region_start:end]
                count_above = np.sum(potential_range > max_threshold)
                if count_above >= min_cds_length:
                    genic_regions.append((genic_region_start, end))
                in_genic_region = False
            genic_region_start = start + step

    # Close the last region
    if in_genic_region:
        potential_range = genic_proba[genic_region_start:seq_length]
        count_above = np.sum(potential_range > max_threshold)
        if count_above >= min_cds_length:
            genic_regions.append((genic_region_start, seq_length))

    return genic_regions


def expand_and_merge_regions(regions, seq_length, buffer_size=100):
    """Expand each region by buffer_size and merge overlapping intervals.

    Args:
        regions: List of (start, end) tuples (inclusive end).
        seq_length: Total sequence length (for clamping).
        buffer_size: Bases to add on each side.

    Returns:
        Merged list of (start, end) tuples, sorted.
    """
    if not regions:
        return []

    # Expand
    expanded = []
    for start, end in regions:
        s = max(0, int(start) - buffer_size)
        e = min(int(seq_length), int(end) + buffer_size)
        if s < e:
            expanded.append((s, e))

    if not expanded:
        return []

    # Sort and merge
    expanded.sort(key=lambda x: x[0])
    merged = [expanded[0]]
    for s, e in expanded[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))

    return merged


def filter_genic_regions(
    gene_probabilities,
    min_avg_threshold=0.1,
    max_threshold=0.5,
    min_cds_length=50,
    buffer_size=100,
    step=50,
):
    """Detect and merge genic regions in one call.

    Returns:
        List of (start, end) tuples for candidate genic regions (inclusive end).
        If no regions found, returns [(0, seq_length)] to decode the full sequence.
    """
    if torch.is_tensor(gene_probabilities):
        seq_length = gene_probabilities.shape[-2]
    else:
        seq_length = gene_probabilities.shape[-2]

    regions = detect_genic_regions(
        gene_probabilities,
        min_avg_threshold=min_avg_threshold,
        max_threshold=max_threshold,
        min_cds_length=min_cds_length,
        step=step,
    )

    if not regions:
        # No genic regions detected; decode the full sequence as fallback
        return [(0, seq_length)]

    merged = expand_and_merge_regions(regions, seq_length, buffer_size=buffer_size)
    return merged if merged else [(0, seq_length)]
