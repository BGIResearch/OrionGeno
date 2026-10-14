"""The shared retry wrapper must also work without the optional GPU backend."""
import sys
from types import SimpleNamespace

import pytest

from oriongeno.inference_runtime import predict_with_memory_backoff


@pytest.mark.parametrize('save', [False, True])
def test_prediction_without_torch_preserves_results_and_options(monkeypatch, save):
    # Fail any attempt to import torch, even in an environment where it exists.
    monkeypatch.setitem(sys.modules, 'torch', None)
    chunks = object()
    outputs = {'gene': object()}
    calls = []

    def predict(inputs, **kwargs):
        assert inputs is chunks
        calls.append(kwargs)
        return outputs

    predictor = SimpleNamespace(get_prediction_outputs=predict)
    options = {'save': True} if save else {}
    assert predict_with_memory_backoff(predictor, chunks, output_gene=True, **options) is outputs
    assert calls == [{'save': save, 'output_gene': True}]


@pytest.mark.parametrize('backend_loaded', [False, True])
def test_non_cuda_errors_propagate_without_retry(monkeypatch, backend_loaded):
    class CudaOOM(RuntimeError):
        pass

    backend = SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=CudaOOM))
    monkeypatch.setitem(sys.modules, 'torch', backend if backend_loaded else None)
    failure = ValueError('invalid prediction input')
    calls = []

    def predict(inputs, **kwargs):
        calls.append(inputs)
        raise failure

    predictor = SimpleNamespace(get_prediction_outputs=predict, hmm=True, hmm_decode_batch=8)
    chunks = object()
    with pytest.raises(ValueError) as caught:
        predict_with_memory_backoff(predictor, chunks)
    assert caught.value is failure
    assert calls == [chunks]
    assert predictor.hmm_decode_batch == 8
