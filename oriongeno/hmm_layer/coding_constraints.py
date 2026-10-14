"""Exact Viterbi repair with spliced-codon and UTR-direction memory.

The checkpoint's labels and scores stay unchanged. Internal state copies carry
the preceding one/two CDS bases through introns, and the side of UTR introns.
An already legal unconstrained optimum needs no second decode. Otherwise we
maximize the same scores over the constrained state graph, without filtering
genes or overwriting their sequences. N and window-initial missing bases remain
unknown; they are never silently treated as A.

A separate noncoding competitor reuses UTR emission scores and never enters
CDS states. Its winning latent states project to public IR labels, so a
noncoding signal need not invent a tiny ORF merely to reach a 3-prime UTR.
Competition is limited to complete short ORFs after coding repair; established
and partial coding models are anchored throughout the competition pass.
"""
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from itertools import product

import numpy as np
import torch

try:
    from numba import njit
except ImportError:
    # Correct (but slower) fallback also applies when the base decoder uses GPU.
    def njit(*args, **kwargs):
        return lambda function: function

from .gene_pred_hmm_topology import expanded_transition_edges

# Memory encoding: 0 = empty; 1..5 = A/C/G/T/N; 6..30 = pairs in
# lexicographic A/C/G/T/N order; 31/32 = 5'/3' UTR. The path validator alone
# uses 33 for a UTR intron whose upstream side lies outside the window.


@njit(cache=True)
def _base(state):
    return 0 if state == 0 else (state - 1) % 19 + 1


@njit(cache=True)
def _advance(src, dst, memory, nucleotide, stops):
    if src == 19 and ((dst == 15 and memory == 32) or (dst == 16 and memory == 31)):
        return -1
    if dst == 4 or dst == 7 or dst == 10 or dst == 12:
        return 1 + nucleotide
    if dst == 5 or dst == 8 or dst == 13:
        first = memory - 1 if 1 <= memory <= 5 else 4
        return 6 + 5 * first + nucleotide
    if dst == 6 or dst == 9 or dst == 11 or dst == 14:
        pair = memory - 6 if 6 <= memory <= 30 else 24
        if dst != 14 and stops[pair // 5, pair % 5, nucleotide]:
            return -1
        return 0
    if dst == 1 or dst == 2 or dst == 3:
        return memory
    if dst == 17:
        return 31 if src == 15 else 32
    if dst == 18 or dst == 19:
        return memory
    return 0


def _memories(base):
    if base in (3, 4, 7, 10, 12):
        return range(1, 6)
    if base in (1, 5, 8, 13):
        return range(6, 31)
    if base in (17, 18, 19):
        return (31, 32)
    return (0,)


@lru_cache(maxsize=16)
def _graph(copies, stop_patterns):
    stops = np.zeros((5, 5, 5), dtype=np.bool_)
    alphabet = dict(zip('ACGT', ((0,), (1,), (2,), (3,))))
    alphabet.update(N=(0, 1, 2, 3), R=(0, 2), Y=(1, 3), W=(0, 3), S=(1, 2),
                    K=(2, 3), M=(0, 1), B=(1, 2, 3), D=(0, 2, 3), H=(0, 1, 3), V=(0, 1, 2))
    for pattern in stop_patterns:
        for codon in product(*(alphabet[c] for c in pattern.upper())):
            stops[codon] = True
    states = [(s, m) for s in range(1 + 19 * copies) for m in _memories(_base(s))]
    index = {state: i for i, state in enumerate(states)}
    labels = np.array([s for s, _ in states], dtype=np.int32)
    initial = np.full((5, len(states)), False)
    edges = expanded_transition_edges(copies)
    allowed = np.zeros((1 + 19 * copies,) * 2, dtype=np.bool_)
    for a, b in edges:
        allowed[a, b] = True
    sources, targets = [], []
    for i, (s, memory) in enumerate(states):
        b = _base(s)
        # Unknown prefix at a window start, but consume the observed first base.
        prior = 30 if b in (6, 9, 11, 14) else 5
        for n in range(5):
            expected = _advance(0, b, prior, n, stops)
            if b in (1, 2, 3):
                expected = {1: 30, 2: 0, 3: 5}[b]
            initial[n, i] = memory == expected or b in (17, 18, 19)
        for a, t in edges:
            if a != s:
                continue
            sources.append(i)
            targets.append([index.get((t, _advance(b, _base(t), memory, n, stops)), -1)
                            for n in range(5)])
    return (stops, labels, initial, np.array(sources, dtype=np.int32),
            np.array(targets, dtype=np.int32).T.copy(), allowed)


@njit(cache=True)
def _legal(path, nucleotides, stops, allowed):
    memory = 33  # unknown UTR side before this window
    first = _base(path[0])
    if first in (1, 5, 8, 13):
        memory = 30 if first == 1 else 26 + nucleotides[0]
    elif first in (3, 4, 7, 10, 12):
        memory = 5 if first == 3 else 1 + nucleotides[0]
    elif first not in (17, 18, 19):
        memory = 0
    for i in range(1, len(path)):
        src, dst = path[i - 1], path[i]
        if not allowed[src, dst]:
            return False
        memory = _advance(_base(src), _base(dst), memory, nucleotides[i], stops)
        if memory < 0:
            return False
    return True


@lru_cache(maxsize=16)
def _competition_graph(copies, stop_patterns):
    """Add a separate noncoding hypothesis, scored by the existing UTR heads.

    This branch cannot enter a CDS state. Its internal states project to IR in
    the public coding annotation. Coding UTRs retain both direction constraints;
    the noncoding branch can finish without inventing a START/STOP unit.
    """
    stops, labels, initial, sources, targets, allowed = _graph(copies, stop_patterns)
    coding_count = len(labels)
    nc_labels, nc_sources, nc_targets = [], [], []
    # Local branch states: exon A/B, and intron entry/body/exit for each side.
    base_labels = (15, 16, 17, 18, 19, 17, 18, 19)
    edges = ((0,0),(0,2),(2,3),(3,3),(3,4),(4,0),(4,1),
             (1,1),(1,5),(5,6),(6,6),(6,7),(7,1))
    for copy in range(copies):
        offset = coding_count + len(nc_labels)
        nc_labels.extend(19 * copy + s for s in base_labels)
        nc_sources.append(0)
        nc_targets.append(offset)
        for src, dst in edges:
            nc_sources.append(offset + src)
            nc_targets.append(offset + dst)
        nc_sources.append(offset + 1)
        nc_targets.append(0)
    nc_labels = np.asarray(nc_labels, dtype=np.int32)
    nc_initial = np.ones((5, len(nc_labels)), dtype=np.bool_)
    return (stops, np.concatenate((labels, nc_labels)),
            np.concatenate((initial, nc_initial), axis=1),
            np.concatenate((sources, np.asarray(nc_sources, dtype=np.int32))),
            np.concatenate((targets, np.tile(np.asarray(nc_targets, dtype=np.int32), (5,1))), axis=1),
            allowed, np.concatenate((labels, np.zeros(len(nc_labels), dtype=np.int32))),
            coding_count)


@njit(cache=True)
def _short_orf_competition_regions(path):
    """Protect established/partial coding paths; expose complete short ORFs.

    60 bp defines the short-ORF review range, not an output length filter.
    In particular a partial long gene at a window edge is never eligible.
    """
    protected = np.zeros(len(path), dtype=np.bool_)
    eligible = False
    start = 0
    while start < len(path):
        if path[start] == 0:
            start += 1
            continue
        end, coding, starts, stops = start, 0, 0, 0
        while end < len(path) and path[end] != 0:
            base = _base(path[end])
            coding += int(4 <= base <= 14)
            starts += int(base == 7)
            stops += int(base == 14)
            end += 1
        short_complete = 0 < coding < 60 and starts == 1 and stops == 1
        if short_complete:
            eligible = True
        elif coding:
            protected[start:end] = True
        start = end
    return eligible, protected


@njit(cache=True, nogil=True)
def _decode(
    emit,
    transition,
    init,
    nucleotides,
    labels,
    initial,
    sources,
    targets,
    output_labels=None,
    final_allowed=None,
    noncoding_start=0,
    protected=None,
    anchor_path=None,
):
    length, count = len(nucleotides), len(labels)
    back = np.zeros((length, count), dtype=np.uint16)
    scores = np.full(count, -np.inf, dtype=np.float32)
    nxt = np.empty(count, dtype=np.float32)
    # Expanded states share public labels: evaluate each emission log once.
    logemit = np.empty(emit.shape[1], dtype=np.float64)
    edge_scores = np.empty(targets.shape, dtype=transition.dtype)
    for n in range(targets.shape[0]):
        for e in range(len(sources)):
            dst = targets[n, e]
            edge_scores[n, e] = transition[labels[sources[e]], labels[dst]] if dst >= 0 else -np.inf
    for s in range(count):
        label = labels[s]
        if initial[nucleotides[0], s] and init[label] > 0 and emit[0, label] > 0:
            scores[s] = np.log(init[label]) + np.log(max(emit[0, label], 1.17549435e-38))
        if protected is not None and protected[0] and s >= noncoding_start:
            scores[s] = -np.inf
        if anchor_path is not None and protected[0] and labels[s] != anchor_path[0]:
            scores[s] = -np.inf
    for i in range(1, length):
        nxt.fill(-np.inf)
        n = nucleotides[i]
        for e in range(len(sources)):
            src, dst = sources[e], targets[n, e]
            if dst < 0:
                continue
            value = scores[src] + edge_scores[n, e]
            if value > nxt[dst]:
                nxt[dst] = value
                back[i, dst] = src
        for label in range(emit.shape[1]):
            value = emit[i, label]
            logemit[label] = np.log(max(value, 1.17549435e-38)) if value > 0 else -np.inf
        for s in range(count):
            nxt[s] += logemit[labels[s]]
            if protected is not None and protected[i] and s >= noncoding_start:
                nxt[s] = -np.inf
            if anchor_path is not None and protected[i] and labels[s] != anchor_path[i]:
                nxt[s] = -np.inf
        # Preserve long-window precision; unreachable prefixes remain -inf.
        offset = np.max(nxt)
        if np.isfinite(offset):
            nxt -= offset
        scores, nxt = nxt, scores
    if final_allowed is not None:
        for s in range(count):
            if not final_allowed[s]:
                scores[s] = -np.inf
    state = np.argmax(scores)
    if not np.isfinite(scores[state]):
        raise ValueError("No legal coding HMM path satisfies the emissions and endpoint hints")
    result = np.empty(length, dtype=np.int32)
    for i in range(length - 1, -1, -1):
        result[i] = labels[state] if output_labels is None else output_labels[state]
        state = back[i, state]
    return result


def repair_coding_paths(
    paths, nucleotides, cell, stacked_inputs, end_hints, copies, stop_codons, emission_probs=None
):
    graph = _competition_graph(copies, tuple(p for p, weight in stop_codons if weight > 0))
    stops, labels, initial, sources, targets, allowed, public_labels, coding_count = graph
    final_allowed = None
    if end_hints is not None:
        # Endpoint hints name public states. Do not let a latent noncoding
        # state consume a UTR hint and then project that endpoint to IR.
        initial = initial.copy()
        initial[:, coding_count:] = False
        final_allowed = np.arange(len(labels)) < coding_count
    if len(labels) >= 65536:
        raise ValueError("Too many expanded coding states for uint16 traceback")
    bases = nucleotides.detach().cpu().numpy()
    # Mixed/unknown encodings stay unknown, including all-zero padding.
    nuc = np.where(
        (bases[..., :4].max(-1) == 1) & (bases[..., 4] == 0), bases[..., :4].argmax(-1), 4
    ).astype(np.int32)
    original = paths.detach().cpu().numpy()
    bad = [
        (m, b)
        for m in range(len(original))
        for b in range(original.shape[1])
        if not _legal(original[m, b], nuc[b], stops, allowed)
    ]
    if not bad:
        return paths
    result = original.copy()
    transition = cell.log_A_dense.detach().float().cpu().numpy()
    init = cell.init_dist.permute(1, 0, 2).detach().float().cpu().numpy()
    # Share immutable emissions across both decoders. Legacy callers may still
    # request lazy per-item computation instead of retaining a batch tensor.
    emissions = None if emission_probs is None else emission_probs.detach().float().cpu().numpy()
    strict = _graph(copies, tuple(p for p, weight in stop_codons if weight > 0))

    def repair_item(b):
        if emissions is None:
            hints = None if end_hints is None else end_hints[:, b : b + 1]
            with torch.inference_mode():
                emit = (
                    cell.emission_probs(
                        stacked_inputs[:, b : b + 1], end_hints=hints, training=False
                    )
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
        else:
            emit = emissions[:, b : b + 1]
        for m, item in bad:
            if item != b:
                continue
            result[m, b] = _decode(emit[m, 0], transition[m], init[m, 0], nuc[b], *strict[1:5])
            eligible, protected = _short_orf_competition_regions(result[m, b])
            if eligible:
                result[m, b] = _decode(
                    emit[m, 0],
                    transition[m],
                    init[m, 0],
                    nuc[b],
                    labels,
                    initial,
                    sources,
                    targets,
                    public_labels,
                    final_allowed,
                    coding_count,
                    protected,
                    result[m, b],
                )
            if not _legal(result[m, b], nuc[b], stops, allowed):
                raise RuntimeError("Constrained HMM traceback failed its coding audit")

    items = sorted({b for _, b in bad})
    requested = max(1, int(os.environ.get("ORIONGENO_HMM_REPAIR_WORKERS", "4")))
    budget = max(1, int(os.environ.get("ORIONGENO_HMM_REPAIR_MEMORY_MB", "1024"))) * 1024**2
    # Each worker has one uint16 traceback, plus path/mask/work buffers.
    per_worker = nuc.shape[1] * (2 * len(labels) + 32)
    workers = min(requested, len(items), max(1, budget // max(1, per_worker)))
    # Fallback emission computation can touch mutable cell state; keep it serial.
    if workers == 1 or emissions is None:
        for b in items:
            repair_item(b)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(repair_item, items))
    logging.info("Re-decoded %d HMM paths with spliced-codon/UTR constraints", len(bad))
    return torch.as_tensor(result, device=paths.device, dtype=paths.dtype)
