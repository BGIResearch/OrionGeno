import json
from types import SimpleNamespace

import numpy as np
import pytest

from oriongeno.window_layout import (
    compact_core_rows, compact_model_length, compact_prediction_groups,
    model_input_alignment,
)


def test_alignment_comes_from_actual_pyramid(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps({'settings': {'feature_scales': [1, 2, 4, 8]}}))
    assert model_input_alignment(tmp_path) == 16
    assert compact_model_length(10_000, 105_600, 40_000, 16, 'compact-short', 1) == 90112
    assert compact_model_length(10_000, 720_000, 40_000, 16, 'compact-short', 640001) == 720000
    assert compact_model_length(10_000, 720_000, 40_000, 16, 'compact', 640001) == 90112


def test_compaction_crops_opposite_ends_on_opposite_strands():
    window = np.repeat((np.arange(8320) % 20)[:, None], 5, axis=1).astype(np.int16)
    items = [dict(strand='+', x_data=window[None], coords=[['seq', '+', 1, 700]]),
             dict(strand='-', x_data=window[::-1][None], coords=[['seq', '-', 1, 700]])]
    calls = []

    class Predictor:
        def adapt_batch_size(self, length):
            pass

        def get_prediction_outputs(self, chunks, **kwargs):
            calls.append(chunks.shape)
            return {'gene': chunks[:, :, 0], 'repeat': chunks[:, :, 0] % 2}

    context = SimpleNamespace(flank_size=64, window_alignment=16, window_sizing='compact-short',
                              output_gene=True, output_repeat=True)
    compact_prediction_groups(Predictor(), items, context)
    assert calls == [(2, 4096, 5)]
    plus = compact_core_rows(items[0]['prediction_outputs']['gene'][:, 64:-64], items[0], 64)[0]
    minus = compact_core_rows(items[1]['prediction_outputs']['gene'][:, 64:-64], items[1], 64)[0]
    np.testing.assert_array_equal(plus, minus[::-1])
    np.testing.assert_array_equal(plus[:700], window[64:764, 0])
    assert items[0]['coords'] == [['seq', '+', 1, 700]]


@pytest.mark.parametrize('strand', ['+', '-'])
def test_unequal_boundary_windows_are_repredicted_without_coordinate_shift(monkeypatch, strand):
    pytest.importorskip('torch')
    from oriongeno.prediction_annotator import PredictionAnnotator
    predictor = PredictionAnnotator(hmm=True, batch_size=1)
    predictor.model_input_alignment = 8
    labels = [np.full(6, 7, dtype=np.int16), np.full(11, 7, dtype=np.int16)]
    chunks = [np.eye(5, dtype=np.uint8)[np.full(len(row), 4)] for row in labels]
    coords = [['seq', strand, 1, 6], ['seq', strand, 7, 17]]
    if strand == '-':
        labels, chunks, coords = labels[::-1], chunks[::-1], coords[::-1]
    seen = []

    def predict(inputs, **kwargs):
        seen.append(inputs.shape)
        assert np.all(inputs[0, 17:, 4] == 1)
        return np.zeros(inputs.shape[:2], dtype=np.int16)

    monkeypatch.setattr(predictor, 'get_predictions', predict)
    annotation, count = predictor.create_gene_annotation(labels, coords, chunks, strand=strand)
    assert seen == [(1, 24, 5)]
    assert count == 0
    assert not annotation.transcripts
    assert all(np.all(row == 0) for row in labels)
    assert predictor.boundary_checks[0]['left_end'] == 6
    assert predictor.boundary_checks[0]['right_start'] == 7
    assert predictor.boundary_checks[0]['status'] == 'joined'


def test_boundary_buckets_restore_input_order(monkeypatch):
    pytest.importorskip('torch')
    from oriongeno.prediction_annotator import PredictionAnnotator
    predictor = PredictionAnnotator()
    predictor.model_input_alignment = 8
    windows = [np.full((length, 5), index, dtype=np.int16)
               for index, length in enumerate([9, 5, 13])]
    seen = []

    def predict(inputs, **kwargs):
        seen.append(inputs.shape)
        return inputs[:, :, 0]

    monkeypatch.setattr(predictor, 'get_predictions', predict)
    results = predictor._predict_boundary_windows(windows, None, 1)
    assert seen == [(2, 16, 5), (1, 8, 5)]
    for index, row in enumerate(results):
        np.testing.assert_array_equal(row, windows[index][:, 0])
