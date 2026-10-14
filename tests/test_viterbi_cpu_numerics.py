"""Numerical regressions for full-window CPU Viterbi decoding."""
import itertools

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("numba")
from oriongeno.hmm_layer.viterbi_cpu import viterbi_decode_cpu  # noqa: E402


def run_decoder(emissions, transitions, initial):
    return viterbi_decode_cpu(
        torch.as_tensor(emissions, dtype=torch.float32).unsqueeze(0),
        torch.as_tensor(transitions, dtype=torch.float32).unsqueeze(0),
        torch.as_tensor(initial, dtype=torch.float32).reshape(1, 1, -1),
    )[0].cpu().numpy()


def safe_log64(values):
    values = np.asarray(values, dtype=np.float32).astype(np.float64)
    return np.where(values == 0, -1000., np.log(np.maximum(values, np.finfo(np.float32).tiny)))


@pytest.mark.parametrize("scale", [1., 1e-3])
def test_long_window_retains_small_late_evidence(scale):
    # Both constant paths tie for 99 kb, then state 1 is strictly better.
    # A switch costs 1000, so the unique optimal path is state 1 throughout.
    emission = np.full((1, 100000, 2), 0.001 * scale, dtype=np.float32)
    emission[0, 99000:, 1] = 0.001001 * scale
    path = run_decoder(emission, [[0., -1000.], [-1000., 0.]], [.5, .5])
    np.testing.assert_array_equal(path, np.ones_like(path))


def test_small_batched_paths_match_exhaustive_float64_scores():
    rng = np.random.default_rng(20260920)
    emission = rng.uniform(.01, .9, size=(3, 7, 3)).astype(np.float32)
    emission[0, 2, 1] = 0.
    transition = np.log(np.array([[.8, .15, .05], [.2, .6, .2], [.1, .2, .7]], dtype=np.float32))
    initial = np.array([.5, .3, .2], dtype=np.float32)
    actual = run_decoder(emission, transition, initial)
    for batch in range(3):
        loge = safe_log64(emission[batch])
        loginit = safe_log64(initial)
        candidates = np.array(list(itertools.product(range(3), repeat=7)))
        scores = loginit[candidates[:, 0]] + loge[np.arange(7), candidates].sum(axis=1)
        scores += transition.astype(np.float64)[candidates[:, :-1], candidates[:, 1:]].sum(axis=1)
        expected = candidates[np.argmax(scores)]
        np.testing.assert_array_equal(actual[batch], expected)


@pytest.mark.parametrize("length", [1, 1000])
def test_equal_scores_keep_first_state(length):
    paths = run_decoder(np.full((2, length, 3), .125, dtype=np.float32), np.zeros((3, 3)), [1 / 3] * 3)
    np.testing.assert_array_equal(paths, np.zeros_like(paths))


def test_coding_redecoder_retains_small_late_evidence():
    from oriongeno.hmm_layer.coding_constraints import _decode

    emission = np.full((100000, 2), .001, dtype=np.float32)
    emission[99000:, 1] = .001001
    path = _decode(
        emission, np.array([[0., -1000.], [-1000., 0.]], dtype=np.float32),
        np.array([.5, .5], dtype=np.float32), np.zeros(100000, dtype=np.int32),
        np.array([0, 1], dtype=np.int32), np.ones((5, 2), dtype=np.bool_),
        np.array([0, 1], dtype=np.int32), np.tile(np.array([0, 1], dtype=np.int32), (5, 1)),
    )
    np.testing.assert_array_equal(path, np.ones_like(path))


def test_coding_redecoder_rejects_unreachable_path():
    from oriongeno.hmm_layer.coding_constraints import _decode

    with pytest.raises(ValueError, match="No legal coding HMM path"):
        _decode(
            np.zeros((3, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float32),
            np.array([.5, .5], dtype=np.float32), np.zeros(3, dtype=np.int32),
            np.array([0, 1], dtype=np.int32), np.ones((5, 2), dtype=np.bool_),
            np.array([0, 1], dtype=np.int32), np.tile(np.array([0, 1], dtype=np.int32), (5, 1)),
        )
