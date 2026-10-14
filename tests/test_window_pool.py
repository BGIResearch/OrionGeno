"""Exercise the real process transport without requiring CUDA in CI."""
import os
import time

import numpy as np
import pytest

from oriongeno.window_pool import (
    WindowPool, predict_with_memory_backoff, window_slices, worker_environment,
)


def fake_worker(connection, device, config, environment, log_path):
    if config.get('startup_error'):
        connection.send({'error': 'initialization failed'})
        return
    connection.send({'ready': True, 'batch_size': 1})
    while True:
        try:
            message = connection.recv()
        except EOFError:
            return
        if message is None:
            return
        if config.get('crash'):
            os._exit(9)
        if config.get('error'):
            connection.send({'error': 'inference failed'})
            return
        # Deliberately let later jobs finish first on the other worker.
        if device == 'a':
            time.sleep(.04)
        chunks = message['chunks']
        outputs = {}
        if message['output_gene']:
            outputs['gene'] = chunks[:, :, 0].copy()
        if message['output_repeat']:
            outputs['repeat'] = (chunks[:, :, 0] % 2).astype(np.int8)
        connection.send({'id': message['id'], 'outputs': outputs, 'seconds': .01})


@pytest.mark.parametrize('count,workers', [(1, 8), (96, 2), (100, 3), (380, 8)])
def test_slices_cover_exactly_once_and_balance_tail(count, workers):
    slices = window_slices(count, workers)
    assert [i for start, end in slices for i in range(start, end)] == list(range(count))
    sizes = [end - start for start, end in slices]
    assert max(sizes) <= 32
    assert max(sizes) - min(sizes) <= 1


def test_process_pool_preserves_order_heads_and_reuses_workers(tmp_path):
    chunks = np.broadcast_to(np.arange(11, dtype=np.int16)[:, None, None], (11, 3, 5))
    pool = WindowPool(['a', 'b'], {}, tmp_path, task_size=3, worker_target=fake_worker)
    try:
        processes = [worker['process'] for worker in pool.workers]
        outputs = pool.predict(chunks, output_repeat=True)
        np.testing.assert_array_equal(outputs['gene'], chunks[:, :, 0])
        np.testing.assert_array_equal(outputs['repeat'], chunks[:, :, 0] % 2)
        np.testing.assert_array_equal(pool.predict_gene(chunks[:1]), chunks[:1, :, 0])
        assert all(process.is_alive() for process in processes)
    finally:
        pool.close()
    assert not any(process.is_alive() for process in processes)
    assert all(process.exitcode == 0 for process in processes)


@pytest.mark.parametrize('failure', ['startup_error', 'error', 'crash'])
def test_worker_failure_propagates_and_reaps_children(tmp_path, failure):
    pool = None
    with pytest.raises(RuntimeError, match='failed|Lost|exited'):
        pool = WindowPool(['a', 'b'], {failure: True}, tmp_path, worker_target=fake_worker)
        pool.predict(np.zeros((4, 3, 5)), output_repeat=True)
    if pool is not None:
        assert all(not worker['process'].is_alive() for worker in pool.workers)


def test_worker_defaults_share_resources_and_respect_overrides():
    env = worker_environment(4, {}, cpu_count=8)
    assert env['ORIONGENO_TORCH_THREADS'] == '1'
    assert env['ORIONGENO_NUMBA_THREADS'] == '1'
    assert env['ORIONGENO_HMM_REPAIR_WORKERS'] == '1'
    env = worker_environment(8, {}, cpu_count=64)
    assert env['ORIONGENO_HMM_REPAIR_MEMORY_MB'] == '512'
    env = worker_environment(4, {'OMP_NUM_THREADS': '7', 'NUMBA_NUM_THREADS': '6',
                                'ORIONGENO_HMM_REPAIR_WORKERS': '2'}, cpu_count=8)
    assert 'ORIONGENO_TORCH_THREADS' not in env
    assert 'ORIONGENO_NUMBA_THREADS' not in env
    assert env['ORIONGENO_HMM_REPAIR_WORKERS'] == '2'


@pytest.mark.parametrize('save', [False, True])
def test_task_oom_backoff_retries_same_inputs_and_keeps_smaller_group(monkeypatch, save):
    torch = pytest.importorskip('torch')
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    calls = []
    chunks = np.zeros((3, 5, 5))

    class Predictor:
        hmm = True
        hmm_decode_batch = 16

        def get_prediction_outputs(self, inputs, **kwargs):
            assert inputs is chunks
            assert kwargs == {'save': save, 'output_gene': True}
            calls.append(self.hmm_decode_batch)
            if self.hmm_decode_batch > 4:
                raise torch.cuda.OutOfMemoryError('simulated overlap allocation')
            return {'gene': inputs[:, :, 0]}

    predictor = Predictor()
    outputs = predict_with_memory_backoff(predictor, chunks, save=save, output_gene=True)
    np.testing.assert_array_equal(outputs['gene'], chunks[:, :, 0])
    assert calls == [16, 8, 4]
    assert predictor.hmm_decode_batch == 4


def test_backend_policy_is_opt_in_and_legacy_policy_can_be_restored(monkeypatch):
    torch = pytest.importorskip('torch')
    from oriongeno.inference_runtime import configure_inference_backend
    previous = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    try:
        monkeypatch.delenv('ORIONGENO_DETERMINISTIC_CUDNN', raising=False)
        configure_inference_backend()
        assert (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic) == previous
        monkeypatch.setenv('ORIONGENO_DETERMINISTIC_CUDNN', '1')
        configure_inference_backend()
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark
        monkeypatch.setenv('ORIONGENO_DETERMINISTIC_CUDNN', '0')
        configure_inference_backend()
        assert not torch.backends.cudnn.deterministic
    finally:
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = previous
