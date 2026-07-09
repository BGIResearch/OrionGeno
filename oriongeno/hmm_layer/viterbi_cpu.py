"""Numba-JIT CPU Viterbi decoder for the gene-prediction HMM.

This is an optional drop-in replacement for the GPU Viterbi recurrence in
``Viterbi.py``. The GPU path decodes with a per-timestep Python loop that
launches several CUDA kernels on tiny ``(batch, q)`` tensors; for the 20-state
gene HMM the real arithmetic is microseconds while kernel-launch latency
dominates (see docs/INFERENCE_OPTIMIZATION.md). Compiling the full-sequence
recurrence to native code removes that launch overhead entirely and lets us
parallelize trivially across the batch with ``prange``.
Each batch item is decoded independently.

Enabled by default; force the GPU path with ``ORIONGENO_HMM_CPU_VITERBI`` set to
one of {0,false,no,off}. Only the homogeneous,
single-model (``num_models==1``) case is handled; the caller falls back to the
GPU path otherwise. The decoder runs the full sequence in one pass, so the
``parallel_factor`` chunking that the GPU path needs is irrelevant here.

Numerics mirror ``Viterbi.safe_log``: zero probabilities map to ``-1e3`` and
all other values use ``log(clamp(x, tiny))`` with ``tiny`` the smallest
positive float32. Transition scores (``log_A_dense``) are already in log space
and are consumed verbatim. ``argmax`` ties resolve to the first (lowest-index)
state, matching ``torch.max`` / ``torch.argmax`` tie-breaking.
"""

import numpy as np
from numba import njit, prange

# Matches Viterbi.safe_log: log_zero_val for exact zeros, and the float32 tiny
# floor used by torch.clamp(min=torch.finfo(float32).tiny).
_LOG_ZERO_VAL = -1.0e3
_TINY_F32 = float(np.finfo(np.float32).tiny)


@njit(cache=True, fastmath=False, inline="always")
def _safe_log_scalar(x):
    """Scalar mirror of Viterbi.safe_log for a single probability."""
    if x == 0.0:
        return _LOG_ZERO_VAL
    if x < _TINY_F32:
        x = _TINY_F32
    return np.log(x)


@njit(cache=True, fastmath=False)
def _viterbi_one(emit, log_A, init, out):
    """Decode one sequence in place.

    Args:
        emit: (L, q) float32 emission probabilities (NOT log-space).
        log_A: (q, q) float32 log transition scores, indexed [from, to].
        init: (q,) float32 initial distribution (NOT log-space).
        out: (L,) int32 output buffer for the decoded state path.
    """
    L = emit.shape[0]
    q = emit.shape[1]

    # dp[s] = best log-prob of any path ending in state s at the current step.
    dp = np.empty(q, dtype=np.float32)
    dp_next = np.empty(q, dtype=np.float32)
    # backpointer[i, s] = best predecessor state for state s at step i (i>=1).
    backptr = np.empty((L, q), dtype=np.int32)

    # Step 0: init * first emission, in log space.
    for s in range(q):
        dp[s] = _safe_log_scalar(init[s]) + _safe_log_scalar(emit[0, s])
        backptr[0, s] = 0

    # Forward recurrence. For each target state, pick the predecessor that
    # maximizes (dp_prev[from] + log_A[from, to]); ties keep the first `from`.
    for i in range(1, L):
        for to in range(q):
            best_from = 0
            best_val = dp[0] + log_A[0, to]
            for fr in range(1, q):
                cand = dp[fr] + log_A[fr, to]
                if cand > best_val:
                    best_val = cand
                    best_from = fr
            dp_next[to] = best_val + _safe_log_scalar(emit[i, to])
            backptr[i, to] = best_from
        for s in range(q):
            dp[s] = dp_next[s]

    # Terminate: best final state (first max on tie).
    best_state = 0
    best_val = dp[0]
    for s in range(1, q):
        if dp[s] > best_val:
            best_val = dp[s]
            best_state = s

    # Backtrack.
    out[L - 1] = best_state
    for i in range(L - 1, 0, -1):
        best_state = backptr[i, best_state]
        out[i - 1] = best_state


@njit(cache=True, parallel=True, fastmath=False)
def _viterbi_batch(emit, log_A, init, out):
    """Decode a batch of sequences, one prange iteration per sequence.

    Args:
        emit: (B, L, q) float32 emission probabilities (NOT log-space).
        log_A: (q, q) float32 log transition scores, indexed [from, to].
        init: (q,) float32 initial distribution (NOT log-space).
        out: (B, L) int32 output buffer for decoded paths.
    """
    B = emit.shape[0]
    for b in prange(B):
        _viterbi_one(emit[b], log_A, init, out[b])


def viterbi_decode_cpu(emission_probs, log_A_dense, init_dist):
    """Full-sequence CPU Viterbi for the single-model gene HMM.

    Args:
        emission_probs: torch.Tensor (1, B, L, q), emission probabilities.
        log_A_dense: torch.Tensor (1, q, q), log-space transitions [from, to].
        init_dist: torch.Tensor (1, 1, q), initial distribution.

    Returns:
        torch.Tensor (1, B, L) int32 decoded state paths, on the input device.
    """
    import torch

    assert emission_probs.dim() == 4 and emission_probs.size(0) == 1, (
        "CPU Viterbi expects emission_probs shape (1, B, L, q); "
        f"got {tuple(emission_probs.shape)}."
    )
    device = emission_probs.device

    emit = emission_probs[0].detach().to("cpu", torch.float32).contiguous().numpy()
    log_A = log_A_dense[0].detach().to("cpu", torch.float32).contiguous().numpy()
    init = init_dist.reshape(-1).detach().to("cpu", torch.float32).contiguous().numpy()

    B, L, q = emit.shape
    assert log_A.shape == (q, q), f"log_A shape {log_A.shape} != ({q}, {q})."
    assert init.shape[0] == q, f"init shape {init.shape} != ({q},)."

    out = np.empty((B, L), dtype=np.int32)
    _viterbi_batch(emit, log_A, init, out)

    return torch.from_numpy(out).to(device).unsqueeze(0)
