"""Correctness contracts for bounded inference concurrency."""
import threading

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('numba')

from oriongeno.prediction_annotator import PredictionAnnotator  # noqa: E402
from oriongeno.hmm_layer import GenePredHMMLayer, Viterbi  # noqa: E402
from test_hmm_coding_constraints import CASES, bases  # noqa: E402


def test_emissions_computed_once_and_parallel_repair_matches_serial(monkeypatch):
    sequence, path = CASES[1]
    probs = torch.full((4, len(path), 20), .0001)
    probs[:, torch.arange(len(path)), torch.tensor(path)] = .99
    ids = torch.tensor(np.tile(bases(sequence), (4, 1)), dtype=torch.long)
    nuc = torch.nn.functional.one_hot(ids, 5).float()
    monkeypatch.setattr(Viterbi, 'viterbi', lambda *a, **k: torch.tensor([[path] * 4]))
    hmm = GenePredHMMLayer()
    emitter = hmm.cell.emission_probs
    calls = []

    def count(*args, **kwargs):
        calls.append(1)
        return emitter(*args, **kwargs)

    monkeypatch.setattr(hmm.cell, 'emission_probs', count)
    with torch.inference_mode():
        monkeypatch.setenv('ORIONGENO_HMM_REPAIR_WORKERS', '1')
        serial = hmm.viterbi(probs, nuc)
        assert len(calls) == 1
        monkeypatch.setenv('ORIONGENO_HMM_REPAIR_WORKERS', '4')
        parallel = hmm.viterbi(probs, nuc)
        assert len(calls) == 2
    torch.testing.assert_close(serial, parallel, rtol=0, atol=0)
    assert not np.array_equal(serial[0].numpy(), path)


def test_pipeline_overlaps_and_keeps_order_with_tail_and_repeat(monkeypatch):
    predictor = PredictionAnnotator(hmm=True, hmm_decode_batch=2)
    inputs = np.arange(7).reshape(7, 1, 1)
    decoder_started = threading.Event()
    next_forward = threading.Event()
    forward_sizes = []

    def forward(chunks, **kwargs):
        forward_sizes.append(len(chunks))
        if chunks[0, 0, 0] == 2:
            assert decoder_started.wait(10), 'first decode did not start'
            next_forward.set()
        assert kwargs['save'] is False
        return {'gene': chunks[:, :, 0].copy(), 'repeat': chunks[:, :, 0].copy() + 20}

    def decode(chunks, probabilities, **kwargs):
        assert torch.is_inference_mode_enabled()
        if chunks[0, 0, 0] == 0:
            decoder_started.set()
            assert next_forward.wait(10), 'forward did not overlap decode'
        np.testing.assert_array_equal(chunks[:, :, 0], probabilities)
        return probabilities + 10

    monkeypatch.setattr(predictor, 'sequence_prediction', forward)
    monkeypatch.setattr(predictor, 'hmm_predictions_filtered', decode)
    result = predictor._pipelined_prediction_outputs(inputs, 1, True)
    assert forward_sizes == [2, 2, 2, 1]
    np.testing.assert_array_equal(result['gene'].ravel(), np.arange(7) + 10)
    np.testing.assert_array_equal(result['repeat'].ravel(), np.arange(7) + 20)


def test_pipeline_propagates_worker_failure(monkeypatch):
    predictor = PredictionAnnotator(hmm=True, hmm_decode_batch=2)
    calls = []

    def forward(chunks, **kwargs):
        calls.append(1)
        return chunks

    def fail(*args, **kwargs):
        raise ValueError('decode failed')

    monkeypatch.setattr(predictor, 'sequence_prediction', forward)
    monkeypatch.setattr(predictor, 'hmm_predictions_filtered', fail)
    with pytest.raises(ValueError, match='decode failed'):
        predictor._pipelined_prediction_outputs(np.zeros((8, 2, 5)), 1, False)
    assert len(calls) <= 2


def test_pipeline_preserves_forward_batch_reduction(monkeypatch):
    predictor = PredictionAnnotator(hmm=True, batch_size=4, hmm_decode_batch=4)
    batches = []

    def forward(chunks, **kwargs):
        batches.append(kwargs['batch_size'])
        predictor.adapted_batch_size = 1  # As after sequence_prediction retries an OOM.
        return chunks[:, :, 0]

    monkeypatch.setattr(predictor, 'sequence_prediction', forward)
    monkeypatch.setattr(predictor, 'hmm_predictions_filtered', lambda x, p, **kw: p)
    predictor._pipelined_prediction_outputs(np.zeros((9, 2, 5)), 4, False)
    assert batches == [4, 1, 1]


def test_pipeline_respects_cache_oracle_and_disable(monkeypatch):
    predictor = PredictionAnnotator(hmm=True)
    predictor.model_device = torch.device('cuda')
    monkeypatch.setenv('ORIONGENO_HMM_PIPELINE', '1')
    assert predictor._use_hmm_pipeline(None, True)
    assert not predictor._use_hmm_pipeline(np.zeros(1), True)
    assert not predictor._use_hmm_pipeline(None, False)
    predictor.temp_dir = 'cache'
    assert not predictor._use_hmm_pipeline(None, True)
    predictor.temp_dir = None
    monkeypatch.setenv('ORIONGENO_HMM_PIPELINE', '0')
    assert not predictor._use_hmm_pipeline(None, True)


def test_cli_thread_defaults_and_explicit_overrides(monkeypatch):
    import numba
    from oriongeno.inference_runtime import configure_inference_threads
    observed = {}
    for key in ('ORIONGENO_TORCH_THREADS', 'ORIONGENO_NUMBA_THREADS',
                'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(torch, 'set_num_threads', lambda n: observed.update(torch=n))
    monkeypatch.setattr(numba, 'set_num_threads', lambda n: observed.update(numba=n))
    configure_inference_threads()
    assert 1 <= observed['torch'] <= 2
    assert 1 <= observed['numba'] <= 4
    observed.clear()
    monkeypatch.setenv('OMP_NUM_THREADS', '3')
    configure_inference_threads()
    assert 'torch' not in observed
    monkeypatch.setenv('ORIONGENO_TORCH_THREADS', '3')
    monkeypatch.setenv('ORIONGENO_NUMBA_THREADS', '2')
    configure_inference_threads()
    assert observed == {'torch': 3, 'numba': 2}
