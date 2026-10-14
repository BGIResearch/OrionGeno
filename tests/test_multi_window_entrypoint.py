"""The public multi entry point uses the window scheduler without opt-in flags."""
from pathlib import Path
import os
import sys
from types import SimpleNamespace

import pytest

from oriongeno.multi_predict import (
    build_multi_parser, build_prediction_args, cleanup_work_dir,
    prepare_work_dir, run_multi_prediction,
)


def test_default_multi_dispatches_to_windows(monkeypatch):
    from oriongeno import window_pool

    args = build_multi_parser().parse_args([
        '--genome', 'input.fa', '--checkpoint', 'weights', '--output', 'output.gff3',
        '--devices', '0,1',
    ])
    calls = []
    monkeypatch.setattr(window_pool, 'run_window_prediction',
                        lambda options, devices: calls.append((options, devices)))
    run_multi_prediction(args)
    assert calls == [(args, ['0', '1'])]
    assert args.window_sizing == 'legacy'


@pytest.mark.parametrize('value', ['windows', 'records'])
def test_schedule_option_is_removed(value):
    parser = build_multi_parser()
    assert not hasattr(parser.parse_args([]), 'schedule')
    with pytest.raises(SystemExit):
        parser.parse_args(['--schedule', value])
    assert '--schedule' not in parser.format_help()


def test_one_gpu_runs_in_process_with_selected_device_and_no_work_directory(monkeypatch, tmp_path):
    from oriongeno import prediction_runner, window_pool

    monkeypatch.setattr(os, 'environ', dict(os.environ))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        cuda=SimpleNamespace(is_initialized=lambda: False)))
    args = build_multi_parser().parse_args([
        '--genome', str(tmp_path/'input.fa'), '--checkpoint', str(tmp_path/'weights'),
        '--output', str(tmp_path/'output.gff3'), '--devices', '1', '--batch-size', '1',
    ])
    calls = []

    def predict(options, **kwargs):
        assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
        assert kwargs == {}
        calls.append(options)
        return 'finished'

    def unexpected_pool(*args, **kwargs):
        pytest.fail('Single GPU inference must not create a worker pool')

    monkeypatch.setattr(prediction_runner, 'run_prediction', predict)
    monkeypatch.setattr(window_pool, 'WindowPool', unexpected_pool)
    assert run_multi_prediction(args) == 'finished'
    assert len(calls) == 1
    assert calls[0].out == args.output
    assert calls[0].batch_size == 1
    assert not (tmp_path/'output.gff3.windows').exists()


def test_initialized_cuda_keeps_isolated_worker_without_remapping_parent(monkeypatch, tmp_path):
    from oriongeno import prediction_runner, window_pool

    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        cuda=SimpleNamespace(is_initialized=lambda: True)))
    args = build_multi_parser().parse_args([
        '--genome', str(tmp_path/'input.fa'), '--checkpoint', str(tmp_path/'weights'),
        '--output', str(tmp_path/'output.gff3'), '--devices', '1', '--batch-size', '1',
    ])
    events = []

    class Pool:
        batch_size = 1

        def __init__(self, devices, config, work_dir, **kwargs):
            assert devices == ['1']
            events.append('created')

        def close(self):
            events.append('closed')

    def predict(options, session):
        assert isinstance(session['window_pool'], Pool)
        assert os.environ['CUDA_VISIBLE_DEVICES'] == '0,1'
        events.append('predicted')

    monkeypatch.setattr(window_pool, 'WindowPool', Pool)
    monkeypatch.setattr(prediction_runner, 'run_prediction', predict)
    run_multi_prediction(args)
    assert events == ['created', 'predicted', 'closed']
    assert not (tmp_path/'output.gff3.windows').exists()


def test_shared_runner_receives_explicit_zero_and_false_over_env(monkeypatch, tmp_path):
    pytest.importorskip('torch')
    from oriongeno.prediction_runner import build_predict_parser

    monkeypatch.setenv('ORIONGENO_HMM_DECODE_BATCH', '16')
    monkeypatch.setenv('ORIONGENO_PROFILE_HMM', 'true')
    args = build_multi_parser().parse_args([
        '--genome', str(tmp_path/'genome.fa'), '--checkpoint', str(tmp_path/'weights'),
        '--output', str(tmp_path/'out.gff3'), '--hmm-decode-batch', '0',
        '--profile-hmm', 'false', '--window-sizing', 'compact',
        '--hmm-parallel-factor', '5', '--output-repeat', 'true',
    ])
    single = build_predict_parser().parse_args(build_prediction_args(args))
    assert single.genome == args.genome
    assert single.out == args.output
    assert single.hmm_decode_batch == 0
    assert single.profile_hmm is False
    assert single.window_sizing == 'compact'
    assert single.hmm_parallel_factor == 5
    assert single.output_repeat is True


def test_work_directory_requires_ownership_and_preserves_output(tmp_path):
    work = tmp_path/'work'
    work.mkdir()
    retained = work/'shard'
    retained.mkdir()
    (retained/'input.fa').write_text('user data')
    output = tmp_path/'output.gff3'
    output.write_text('prediction')
    with pytest.raises(RuntimeError, match='unrecognized'):
        prepare_work_dir(work, output, False)
    assert (retained/'input.fa').read_text() == 'user data'
    owned = Path(prepare_work_dir(tmp_path/'owned', output, False))
    (owned/'worker.log').write_text('log')
    cleanup_work_dir(owned, output)
    assert not owned.exists()
    assert output.read_text() == 'prediction'
