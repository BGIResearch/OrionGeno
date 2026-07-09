import os
import time

import torch

LOG_ZERO_THRESHOLD = -999.0


def _profile_hmm_enabled():
    return os.environ.get("ORIONGENO_PROFILE_HMM", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _cpu_viterbi_enabled():
    # Default ON: the native-CPU Viterbi is the production path. Force the GPU
    # path with ORIONGENO_HMM_CPU_VITERBI in {0,false,no,off}.
    return os.environ.get("ORIONGENO_HMM_CPU_VITERBI", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


_CPU_VITERBI_AVAILABLE = None


def _cpu_viterbi_available():
    """True if the numba CPU decoder can be imported; cached, warns once."""
    global _CPU_VITERBI_AVAILABLE
    if _CPU_VITERBI_AVAILABLE is None:
        try:
            from .viterbi_cpu import viterbi_decode_cpu  # noqa: F401

            _CPU_VITERBI_AVAILABLE = True
        except Exception as exc:  # numba/llvmlite missing or broken
            _CPU_VITERBI_AVAILABLE = False
            import warnings

            warnings.warn(
                "numba CPU Viterbi unavailable (%s); falling back to the GPU "
                "Viterbi path. Install numba to enable the faster CPU decoder."
                % exc
            )
    return _CPU_VITERBI_AVAILABLE


def _sync_for_timing(tensor):
    if torch.cuda.is_available() and torch.is_tensor(tensor) and tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


def _profile_mark(tensor, enabled):
    if enabled:
        _sync_for_timing(tensor)
    return time.perf_counter()


def _profile_elapsed(start, tensor, enabled):
    if enabled:
        _sync_for_timing(tensor)
    return time.perf_counter() - start


def _sparse_viterbi_enabled(num_states):
    mode = os.environ.get("ORIONGENO_SPARSE_VITERBI", "auto").lower()
    if mode in {
        "0",
        "false",
        "no",
        "off",
    }:
        return False
    if mode in {"1", "true", "yes", "on"}:
        return True
    return num_states >= 64


def _parallel_backpointers_enabled():
    mode = os.environ.get("ORIONGENO_HMM_PARALLEL_BACKPOINTERS", "1").lower()
    return mode not in {
        "0",
        "false",
        "no",
        "off",
    }


def _backpointer_dtype(num_states):
    if num_states <= 255:
        return torch.uint8
    if num_states <= 32767:
        return torch.int16
    return torch.int32


def _make_sparse_transition(transition_matrix):
    if transition_matrix.dim() != 3:
        return None
    if not _sparse_viterbi_enabled(transition_matrix.size(-1)):
        return None

    mask = transition_matrix > LOG_ZERO_THRESHOLD
    if mask.size(0) > 1 and not torch.all(mask == mask[:1]):
        return None

    src_idx, dst_idx = torch.nonzero(mask[0], as_tuple=True)
    if src_idx.numel() == 0 or src_idx.numel() >= mask.size(-1) * mask.size(-1):
        return None

    num_states = transition_matrix.size(-1)
    pred_counts = torch.bincount(dst_idx, minlength=num_states)
    max_pred = int(pred_counts.max().item())
    if max_pred <= 0:
        return None

    pred_src = torch.zeros(
        num_states,
        max_pred,
        dtype=torch.long,
        device=transition_matrix.device,
    )
    pred_scores = torch.full(
        (transition_matrix.size(0), num_states, max_pred),
        torch.finfo(transition_matrix.dtype).min,
        dtype=transition_matrix.dtype,
        device=transition_matrix.device,
    )

    for dst_state in range(num_states):
        edge_mask = dst_idx == dst_state
        count = int(edge_mask.sum().item())
        if count == 0:
            continue
        pred_src[dst_state, :count] = src_idx[edge_mask]
        pred_scores[:, dst_state, :count] = transition_matrix[
            :, src_idx[edge_mask], dst_state
        ]

    return {
        "pred_src": pred_src,
        "pred_scores": pred_scores,
        "num_edges": src_idx.numel(),
        "num_states": num_states,
    }


def viterbi_step_sparse(gamma_prev, emission_probs_i, sparse_transition):
    pred_src = sparse_transition["pred_src"]
    pred_scores = sparse_transition["pred_scores"]

    scores = gamma_prev.index_select(-1, pred_src.reshape(-1))
    scores = scores.view(gamma_prev.shape[:-1] + pred_src.shape)
    score_shape = [pred_scores.size(0)] + [1] * (scores.dim() - 3) + [
        pred_scores.size(1),
        pred_scores.size(2),
    ]
    scores = scores + pred_scores.view(score_shape)
    gamma_next = torch.max(scores, dim=-1).values
    gamma_next += safe_log(emission_probs_i.unsqueeze(2))
    return gamma_next


def safe_log(x, log_zero_val=-1e3):
    """Computes element-wise logarithm with output_i=log_zero_val where x_i=0."""
    if not x.dtype.is_floating_point:
        x = x.to(torch.float32)
    epsilon = torch.finfo(x.dtype).tiny
    log_x = torch.log(torch.clamp(x, min=epsilon))
    zero_mask = (x == 0).to(dtype=log_x.dtype)
    log_zero = torch.as_tensor(log_zero_val, dtype=log_x.dtype, device=log_x.device)
    log_x = (1 - zero_mask) * log_x + zero_mask * log_zero
    return log_x


def viterbi_step(
    gamma_prev, emission_probs_i, transition_matrix, non_homogeneous_mask=None
):
    """Computes one Viterbi dynamic programming step."""
    gamma_next = transition_matrix.unsqueeze(1).unsqueeze(1) + gamma_prev.unsqueeze(-1)

    if non_homogeneous_mask is not None:
        gamma_next += safe_log(non_homogeneous_mask.unsqueeze(2))

    gamma_next, _ = torch.max(gamma_next, dim=-2)
    gamma_next += safe_log(emission_probs_i.unsqueeze(2))

    return gamma_next


def viterbi_dyn_prog(
    emission_probs,
    init,
    transition_matrix,
    use_first_position_emission=True,
    non_homogeneous_mask_func=None,
    sparse_transition=None,
):
    """Run underflow-safe Viterbi decoding for many sequences in parallel."""
    use_sparse = sparse_transition is not None and non_homogeneous_mask_func is None
    gamma_val = safe_log(init).unsqueeze(1)
    gamma_val = gamma_val.to(dtype=transition_matrix.dtype)

    b0 = emission_probs[:, :, 0]
    if use_first_position_emission:
        gamma_val = gamma_val + safe_log(b0).unsqueeze(2)
    else:
        gamma_val = gamma_val + torch.zeros_like(b0).unsqueeze(2)

    L = emission_probs.size(2)
    gamma_list = [gamma_val]

    for i in range(1, L):
        emission_probs_i = emission_probs[:, :, i]
        non_homogeneous_mask = None
        if non_homogeneous_mask_func is not None:
            non_homogeneous_mask = non_homogeneous_mask_func(i)

        if use_sparse:
            gamma_val = viterbi_step_sparse(
                gamma_val, emission_probs_i, sparse_transition
            )
        else:
            gamma_val = viterbi_step(
                gamma_val, emission_probs_i, transition_matrix, non_homogeneous_mask
            )
        gamma_list.append(gamma_val)

    gamma = torch.stack(gamma_list, dim=-2)  # (num_models, b, z, L, q)
    return gamma


def viterbi_dyn_prog_with_backpointers(
    emission_probs,
    init,
    transition_matrix,
    use_first_position_emission=True,
    non_homogeneous_mask_func=None,
):
    """Run local Viterbi DP and store compact backpointers instead of full scores."""
    gamma_val = safe_log(init).unsqueeze(1)
    gamma_val = gamma_val.to(dtype=transition_matrix.dtype)

    b0 = emission_probs[:, :, 0]
    if use_first_position_emission:
        gamma_val = gamma_val + safe_log(b0).unsqueeze(2)
    else:
        gamma_val = gamma_val + torch.zeros_like(b0).unsqueeze(2)

    pointer_dtype = _backpointer_dtype(transition_matrix.size(-1))
    backpointers = []
    for i in range(1, emission_probs.size(2)):
        emission_probs_i = emission_probs[:, :, i]
        scores = transition_matrix.unsqueeze(1).unsqueeze(1) + gamma_val.unsqueeze(-1)
        if non_homogeneous_mask_func is not None:
            scores += safe_log(non_homogeneous_mask_func(i).unsqueeze(2))

        gamma_next, previous_states = torch.max(scores, dim=-2)
        gamma_val = gamma_next + safe_log(emission_probs_i.unsqueeze(2))
        backpointers.append(previous_states.to(dtype=pointer_dtype))

    if backpointers:
        backpointers = torch.stack(backpointers, dim=-2)
    else:
        backpointers = torch.empty(
            gamma_val.shape[:-2] + gamma_val.shape[-2:-1] + (0, gamma_val.size(-1)),
            dtype=pointer_dtype,
            device=gamma_val.device,
        )
    return gamma_val, backpointers


def viterbi_chunk_step(gamma_prev, local_gamma):
    """A variant of the Viterbi step used in parallel Viterbi."""
    gamma_next = local_gamma + gamma_prev.unsqueeze(-1)
    gamma_next, _ = torch.max(gamma_next, dim=-2)
    return gamma_next


def viterbi_chunk_dyn_prog(
    emission_probs,
    init,
    transition_matrix,
    local_gamma,
    non_homogeneous_mask=None,
    sparse_transition=None,
):
    """Computes gamma values at chunk begin and end positions."""
    use_sparse = sparse_transition is not None and non_homogeneous_mask is None
    gamma_val = safe_log(init).unsqueeze(1)
    gamma_val = gamma_val.to(dtype=transition_matrix.dtype)
    b0 = emission_probs[:, :, 0]
    gamma_val = gamma_val + safe_log(b0)

    num_chunks = emission_probs.size(2)


    gamma_list = []
    gamma_list.append(gamma_val)

    gamma_val = viterbi_chunk_step(gamma_val, local_gamma[:, :, 0])
    gamma_list.append(gamma_val)


    for i in range(1, num_chunks):
        if use_sparse:
            gamma_val = viterbi_step_sparse(
                gamma_val.unsqueeze(-2),
                emission_probs[:, :, i],
                sparse_transition,
            )[..., 0, :]
        else:
            gamma_val = viterbi_step(
                gamma_val.unsqueeze(-2),
                emission_probs[:, :, i],
                transition_matrix,
                non_homogeneous_mask,
            )[..., 0, :]
        gamma_list.append(gamma_val)

        gamma_val = viterbi_chunk_step(gamma_val, local_gamma[:, :, i])
        gamma_list.append(gamma_val)

    # Stack and reshape
    gamma = torch.stack(gamma_list, dim=2)  # (num_models, b, 2*num_chunks, q)
    gamma = gamma.view(gamma.size(0), gamma.size(1), num_chunks, 2, gamma.size(-1))

    return gamma


def viterbi_backtracking_step(
    prev_states: torch.Tensor,
    gamma_state: torch.Tensor,
    transition_matrix_transposed: torch.Tensor,
    output_type: torch.dtype,
    non_homogeneous_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Computes a Viterbi backtracking step in parallel for all models and batch elements.

    Args:
        prev_states: Previously decoded states. Shape: (num_model, b, 1)
        gamma_state: Viterbi values of the previously decoded states. Shape: (num_model, b, q)
        transition_matrix_transposed: Transposed logarithmic transition matrices.
                                        Shape (num_models, q, q) or (num_models, b, q, q)
        output_type: Datatype of the output states.
        non_homogeneous_mask: Optional mask of shape (num_models, b, q, q).
    """

    # Number of hidden states.
    Q = transition_matrix_transposed.size(-1)

    prev_states = prev_states.to(torch.int64)
    # Gather expects the index tensor to match all non-gathered dimensions.
    indices_4d = prev_states.unsqueeze(-1)
    indices = indices_4d.expand(-1, -1, -1, Q)

    if non_homogeneous_mask is not None:
        # The mask is transposed so it follows the backtracking transition direction.
        transition_matrix_transposed = transition_matrix_transposed + safe_log(
            non_homogeneous_mask.transpose(-1, -2)
        )

    # Expand model-level matrices to match the batch dimension when needed.
    batch_dims = transition_matrix_transposed.dim() - 2

    if batch_dims == 1:
        B = prev_states.size(1)
        A_expanded = transition_matrix_transposed.unsqueeze(1).expand(-1, B, -1, -1)

        A_prev_states = torch.gather(A_expanded, dim=-2, index=indices)

    else:
        # Batch-specific transition matrices already match the index shape.
        A_prev_states = torch.gather(transition_matrix_transposed, dim=-2, index=indices)

    # Remove the gathered previous-state dimension.
    A_prev_states = A_prev_states.squeeze(-2)

    next_states = torch.argmax(A_prev_states + gamma_state, dim=-1, keepdim=True)
    return next_states.to(dtype=output_type)

def viterbi_backtracking(
    gamma,
    transition_matrix_transposed,
    output_type=torch.int32,
    non_homogeneous_mask_func=None,
):
    """Performs backtracking on Viterbi score tables."""
    cur_states = torch.argmax(gamma[:, :, -1], dim=-1, keepdim=True)
    L = gamma.size(2)
    state_seqs_max_lik = [cur_states]

    for i in range(L - 2, -1, -1):
        cur_states = viterbi_backtracking_step(
            cur_states,
            gamma[:, :, i],
            transition_matrix_transposed,
            output_type,
            (
                non_homogeneous_mask_func(i + 1)
                if non_homogeneous_mask_func is not None
                else None
            ),
        )
        state_seqs_max_lik.append(cur_states)

    state_seqs_max_lik = torch.cat(state_seqs_max_lik[::-1], dim=-1)
    return state_seqs_max_lik


def viterbi_chunk_backtracking(
    gamma,
    local_gamma_end_transposed,
    transition_matrix_transposed,
    output_type=torch.int32,
    non_homogeneous_mask_func=None,
):
    """Performs backtracking on chunk-wise Viterbi score tables."""
    cur_states = torch.argmax(gamma[:, :, -1, 1], dim=-1, keepdim=True)
    num_chunks = gamma.size(2)

    state_seqs_max_lik = [cur_states]

    cur_states = viterbi_backtracking_step(
        cur_states,
        gamma[:, :, -1, 0],
        local_gamma_end_transposed[:, :, -1],
        output_type,
    )
    state_seqs_max_lik.append(cur_states)

    for i in range(1, num_chunks):
        cur_states = viterbi_backtracking_step(
            cur_states,
            gamma[:, :, -1 - i, 1],
            transition_matrix_transposed,
            output_type,
            (
                non_homogeneous_mask_func(num_chunks - i)
                if non_homogeneous_mask_func is not None
                else None
            ),
        )
        state_seqs_max_lik.append(cur_states)

        cur_states = viterbi_backtracking_step(
            cur_states,
            gamma[:, :, -1 - i, 0],
            local_gamma_end_transposed[:, :, -1 - i],
            output_type,
        )
        state_seqs_max_lik.append(cur_states)

    state_seqs_max_lik = torch.cat(state_seqs_max_lik[::-1], dim=-1)
    state_seqs_max_lik = state_seqs_max_lik.view(
        state_seqs_max_lik.size(0), state_seqs_max_lik.size(1), num_chunks, 2
    )
    return state_seqs_max_lik


def viterbi_full_chunk_backtracking(
    viterbi_chunk_borders,
    local_gamma,
    transition_matrix_transposed,
    output_type=torch.int32,
    non_homogeneous_mask_func=None,
):
    """Determines full Viterbi state sequence given optimal chunk endpoints."""
    num_model, b, num_chunks, q, chunk_length, _ = local_gamma.shape
    local_gamma = local_gamma.view(num_model, b * num_chunks, q, chunk_length, q)

    start_states = viterbi_chunk_borders[:, :, :, 0].reshape(
        num_model, b * num_chunks, 1
    )
    end_states = viterbi_chunk_borders[:, :, :, 1].reshape(num_model, b * num_chunks, 1)

    # Gather the correct local_gamma based on start_states

    start_states_expanded = (
        start_states.unsqueeze(-1)
        .unsqueeze(-1)
        .expand(num_model, b * num_chunks, 1, chunk_length, q)
    )
    local_gamma = torch.gather(local_gamma, 2, start_states_expanded).squeeze(2)

    cur_states = end_states
    state_seqs_max_lik = [cur_states]

    for i in range(chunk_length - 2, 0, -1):
        cur_states = viterbi_backtracking_step(
            cur_states,
            local_gamma[:, :, i],
            transition_matrix_transposed,
            output_type,
            (
                non_homogeneous_mask_func(i + 1)
                if non_homogeneous_mask_func is not None
                else None
            ),
        )
        state_seqs_max_lik.append(cur_states)

    state_seqs_max_lik.append(start_states)
    state_seqs_max_lik = torch.cat(state_seqs_max_lik[::-1], dim=-1)
    state_seqs_max_lik = state_seqs_max_lik.view(
        num_model, b, num_chunks * chunk_length
    )

    return state_seqs_max_lik


def viterbi_full_chunk_backtracking_from_backpointers(
    viterbi_chunk_borders,
    local_backpointers,
    output_type=torch.int32,
):
    """Recover full chunk paths from compact local backpointers."""
    num_model, b, num_chunks, num_start_states, backtrack_len, q = local_backpointers.shape
    chunk_length = backtrack_len + 1

    start_states = viterbi_chunk_borders[:, :, :, 0].to(torch.long)
    end_states = viterbi_chunk_borders[:, :, :, 1].to(torch.long)

    start_index = start_states.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    start_index = start_index.expand(
        num_model,
        b,
        num_chunks,
        1,
        backtrack_len,
        q,
    )
    selected_backpointers = torch.gather(
        local_backpointers,
        dim=3,
        index=start_index,
    ).squeeze(3)

    cur_states = end_states.unsqueeze(-1)
    state_seqs_max_lik = [cur_states.to(dtype=output_type)]

    for i in range(backtrack_len - 1, -1, -1):
        current_backpointers = selected_backpointers[:, :, :, i].to(torch.long)
        cur_states = torch.gather(current_backpointers, dim=-1, index=cur_states)
        state_seqs_max_lik.append(cur_states.to(dtype=output_type))

    state_seqs_max_lik = torch.cat(state_seqs_max_lik[::-1], dim=-1)
    state_seqs_max_lik[..., 0] = start_states.to(dtype=output_type)
    state_seqs_max_lik = state_seqs_max_lik.view(
        num_model,
        b,
        num_chunks * chunk_length,
    )
    return state_seqs_max_lik


def viterbi(
    sequences,
    hmm_cell,
    end_hints=None,
    parallel_factor=1,
    return_variables=False,
    non_homogeneous_mask_func=None,
):
    """Computes the most likely sequence of hidden states."""
    profile_hmm = _profile_hmm_enabled()
    profile_total_start = _profile_mark(sequences, profile_hmm)
    profile_times = {}

    if len(sequences.shape) == 3:
        sequences = torch.nn.functional.one_hot(sequences.long(), hmm_cell.dim).to(
            sequences.dtype
        )
    else:
        sequences = sequences.to(sequences.dtype)

    seq_lens = (sequences[..., -1] == 0).to(torch.int32).sum(dim=-1)

    profile_start = _profile_mark(sequences, profile_hmm)
    emission_probs = hmm_cell.emission_probs(
        sequences, end_hints=end_hints, training=False
    )
    profile_times["emission"] = _profile_elapsed(
        profile_start, emission_probs, profile_hmm
    )
    num_model, b, seq_len, q = emission_probs.shape

    # Native-CPU full-sequence Viterbi (default; disable with
    # ORIONGENO_HMM_CPU_VITERBI=0). Compiles the recurrence to machine code,
    # removing the per-step CUDA kernel-launch overhead that dominates the GPU
    # path on the tiny 20-state tensors. Only the homogeneous single-model case
    # is handled; everything else -- and any environment without numba -- falls
    # through to the GPU path below. parallel_factor is irrelevant here (the
    # whole sequence is decoded in one pass).
    if (
        _cpu_viterbi_enabled()
        and num_model == 1
        and non_homogeneous_mask_func is None
        and not return_variables
        and _cpu_viterbi_available()
    ):
        from .viterbi_cpu import viterbi_decode_cpu

        cpu_init = hmm_cell.init_dist.permute(1, 0, 2).to(
            device=emission_probs.device, dtype=emission_probs.dtype
        )
        cpu_A = hmm_cell.log_A_dense.to(
            device=emission_probs.device, dtype=emission_probs.dtype
        )
        profile_start = _profile_mark(emission_probs, profile_hmm)
        viterbi_paths = viterbi_decode_cpu(emission_probs, cpu_A, cpu_init)
        profile_times["local_dp_cpu"] = _profile_elapsed(
            profile_start, viterbi_paths, profile_hmm
        )
        if profile_hmm:
            total = _profile_elapsed(profile_total_start, viterbi_paths, profile_hmm)
            detail = " ".join(
                f"{name}={duration:.3f}s" for name, duration in profile_times.items()
            )
            print(
                "HMM profile (CPU): "
                f"batch={b} seq_len={seq_len} states={q} {detail} total={total:.3f}s"
            )
        return viterbi_paths

    assert (
        seq_len % parallel_factor == 0
    ), f"Sequence length ({seq_len}) must be divisible by parallel_factor ({parallel_factor})."

    chunk_size = seq_len // parallel_factor
    emission_probs = emission_probs.reshape(
        num_model, b * parallel_factor, chunk_size, q
    )

    init_dist = hmm_cell.init_dist.permute(1, 0, 2).to(
        device=emission_probs.device,
        dtype=emission_probs.dtype,
    )
    init = (
        init_dist
        if parallel_factor == 1
        else torch.eye(q, dtype=sequences.dtype, device=emission_probs.device).unsqueeze(
            0
        )
    )
    z = init.size(1)

    A = hmm_cell.log_A_dense.to(device=emission_probs.device, dtype=emission_probs.dtype)
    At = hmm_cell.log_A_dense_t.to(device=emission_probs.device, dtype=emission_probs.dtype)
    sparse_transition = _make_sparse_transition(A)
    use_backpointer_parallel = (
        parallel_factor > 1
        and _parallel_backpointers_enabled()
        and sparse_transition is None
    )

    if non_homogeneous_mask_func is not None:
        from functools import partial

        non_homogeneous_mask_func = partial(
            non_homogeneous_mask_func, seq_lens=seq_lens, hmm_cell=hmm_cell
        )
        sparse_transition = None
        use_backpointer_parallel = False

    profile_start = _profile_mark(emission_probs, profile_hmm)
    if use_backpointer_parallel:
        gamma_local_at_chunk_end, local_backpointers = viterbi_dyn_prog_with_backpointers(
            emission_probs,
            init,
            A,
            use_first_position_emission=False,
            non_homogeneous_mask_func=None,
        )
    else:
        gamma = viterbi_dyn_prog(
            emission_probs,
            init,
            A,
            use_first_position_emission=parallel_factor == 1,
            non_homogeneous_mask_func=non_homogeneous_mask_func,
            sparse_transition=sparse_transition,
        )
    profile_times["local_dp"] = _profile_elapsed(
        profile_start,
        gamma_local_at_chunk_end if use_backpointer_parallel else gamma,
        profile_hmm,
    )

    if not use_backpointer_parallel:
        gamma = gamma.reshape(num_model, b * parallel_factor * z, chunk_size, q)

    if parallel_factor == 1:
        profile_start = _profile_mark(gamma, profile_hmm)
        viterbi_paths = viterbi_backtracking(
            gamma, At, non_homogeneous_mask_func=non_homogeneous_mask_func
        )
        profile_times["backtracking"] = _profile_elapsed(
            profile_start, viterbi_paths, profile_hmm
        )
        variables_out = gamma
    else:
        emission_probs_at_chunk_start = emission_probs[:, :, 0].reshape(
            num_model, b, parallel_factor, q
        )
        if use_backpointer_parallel:
            gamma_local_at_chunk_end = gamma_local_at_chunk_end.reshape(
                num_model, b, parallel_factor, q, q
            )
        else:
            gamma_local_at_chunk_end = gamma[:, :, -1]
            gamma_local_at_chunk_end = gamma_local_at_chunk_end.reshape(
                num_model, b, parallel_factor, q, q
            )

        profile_start = _profile_mark(gamma_local_at_chunk_end, profile_hmm)
        gamma_at_chunk_borders = viterbi_chunk_dyn_prog(
            emission_probs_at_chunk_start,
            init_dist[:, 0],
            A,
            gamma_local_at_chunk_end,
            sparse_transition=sparse_transition,
        )
        profile_times["chunk_border_dp"] = _profile_elapsed(
            profile_start, gamma_at_chunk_borders, profile_hmm
        )

        gamma_local_at_chunk_end = gamma_local_at_chunk_end.permute(0, 1, 2, 4, 3)
        profile_start = _profile_mark(gamma_at_chunk_borders, profile_hmm)
        viterbi_chunk_borders = viterbi_chunk_backtracking(
            gamma_at_chunk_borders, gamma_local_at_chunk_end, At
        )
        profile_times["chunk_border_backtracking"] = _profile_elapsed(
            profile_start, viterbi_chunk_borders, profile_hmm
        )

        profile_start = _profile_mark(gamma_at_chunk_borders, profile_hmm)
        if use_backpointer_parallel:
            local_backpointers = local_backpointers.reshape(
                num_model,
                b,
                parallel_factor,
                z,
                chunk_size - 1,
                q,
            )
            viterbi_paths = viterbi_full_chunk_backtracking_from_backpointers(
                viterbi_chunk_borders,
                local_backpointers,
            )
        else:
            gamma = gamma.reshape(num_model, b, parallel_factor, z, chunk_size, q)
            viterbi_paths = viterbi_full_chunk_backtracking(
                viterbi_chunk_borders, gamma, At
            )
        profile_times["full_backtracking"] = _profile_elapsed(
            profile_start, viterbi_paths, profile_hmm
        )
        variables_out = gamma_at_chunk_borders

    if profile_hmm:
        total = _profile_elapsed(profile_total_start, viterbi_paths, profile_hmm)
        detail = " ".join(
            f"{name}={duration:.3f}s" for name, duration in profile_times.items()
        )
        print(
            "HMM profile: "
            f"batch={b} seq_len={seq_len} states={q} "
            f"parallel_factor={parallel_factor} chunk_size={chunk_size} "
            f"parallel_storage={'backpointers' if use_backpointer_parallel else 'scores'} "
            f"sparse_edges={0 if sparse_transition is None else sparse_transition['num_edges']} "
            f"{detail} total={total:.3f}s"
        )

    if return_variables:
        return viterbi_paths, variables_out
    else:
        return viterbi_paths
