import numpy as np
import pytest

pytest.importorskip('torch')

from oriongeno.data_processing.state_features import gene_state_ranges  # noqa: E402
from oriongeno.data_processing.window_stitching import stitch_window_ranges  # noqa: E402
from oriongeno.data_processing.utr_repair import repair_conflicts, joint_decode  # noqa: E402


def inputs(strand='+', middle=3):
    rows = [[0, 15, 17, 18]] + [[18]*4]*middle + [[18, 19, 16, 0, 7, 5, 6, 4, 5, 14, 0]]
    if strand == '-':
        rows = [row[::-1] for row in rows[::-1]]
    labels = [np.array(row, dtype=np.int16) for row in rows]
    coords, chunks, start = [], [], 1001
    for row in rows:
        coords.append(['chr', strand, start, start+len(row)-1])
        # Distinct genomic-position bases detect reversed-window mistakes.
        nuc = np.eye(5, dtype=np.uint8)[np.arange(start, start+len(row)) % 4]
        chunks.append(nuc if strand == '+' else nuc[::-1])
        start += len(row)
    _, checks = stitch_window_ranges(labels, coords, strand, gene_state_ranges)
    return labels, coords, chunks, checks


@pytest.mark.parametrize('strand', ['+', '-'])
def test_complete_context_crosses_many_windows_and_changes_only_utr(strand):
    labels, coords, chunks, checks = inputs(strand)
    before = np.concatenate(labels).copy()
    def callback(sequence, path, protected, offset):
        expected = np.eye(5, dtype=np.uint8)[np.arange(1001, 1001+len(before)) % 4]
        if strand == '-':
            expected = expected[::-1]
        np.testing.assert_array_equal(sequence, expected)
        assert offset == 0
        assert np.all(np.isin(path[~protected], np.arange(15, 20)))
        repaired = path.copy()
        repaired[~protected] = 0
        return repaired, {'probability_source': 'test_probabilities'}
    audit = repair_conflicts(labels, coords, chunks, strand, checks, callback)
    assert audit[0]['status'] == 'repaired'
    assert audit[0]['windows_spanned'] == 5
    after = np.concatenate(labels)
    coding = (before >= 1) & (before <= 14)
    np.testing.assert_array_equal(after[coding], before[coding])
    assert all(c['status'] == 'joined' for c in stitch_window_ranges(labels, coords, strand, gene_state_ranges)[1])


@pytest.mark.parametrize('failure', ['no_path', 'changed_anchor', 'bad_transition', 'memory'])
def test_failed_proposal_never_changes_input(failure):
    labels, coords, chunks, checks = inputs()
    before = [row.copy() for row in labels]
    def callback(sequence, path, protected, offset):
        if failure == 'no_path':
            raise ValueError('No legal path')
        if failure == 'memory':
            raise MemoryError('budget')
        proposed = path.copy()
        proposed[~protected] = 0
        if failure == 'changed_anchor':
            proposed[0] = 15
        else:
            proposed[np.flatnonzero(~protected)[0]] = 19
        return proposed, {}
    audit = repair_conflicts(labels, coords, chunks, '+', checks, callback)
    assert audit[0]['status'] == 'fallback_split'
    assert audit[0]['reason']
    for a, b in zip(labels, before):
        np.testing.assert_array_equal(a, b)


def test_no_conflict_does_not_call_provider():
    def fail(*args):
        pytest.fail('Provider invoked without a UTR conflict')
    assert repair_conflicts([], [], [], '+', [], fail) == []


def test_budget_falls_back_before_model_forward(monkeypatch):
    monkeypatch.setenv('ORIONGENO_UTR_REPAIR_MAX_BASES', '2')
    labels, coords, chunks, checks = inputs()
    audit = repair_conflicts(labels, coords, chunks, '+', checks, lambda *a: pytest.fail('forward'))
    assert audit[0]['status'] == 'fallback_split'
    assert 'budget' in audit[0]['reason']


def toy_predictor():
    import torch
    from types import SimpleNamespace
    from oriongeno.hmm_layer.gene_pred_hmm_topology import BASE_GENE_PRED_TRANSITIONS
    transition = torch.full((1, 20, 20), -torch.inf)
    for a, targets in BASE_GENE_PRED_TRANSITIONS.items():
        for b in targets:
            transition[0, a, b] = 0
    cell = SimpleNamespace(log_A_dense=transition, init_dist=torch.ones(1, 1, 20)/20,
                           recurrent_init=lambda: None,
                           emission_probs=lambda data, training: data)
    layer = SimpleNamespace(cell=cell, num_models=1, num_copies=1, stop_codons=[('TAA', 1)],
                            concat_inputs=lambda probs, nuc: probs[None])
    return SimpleNamespace(gene_pred_hmm_layer=layer, upper_only=True, _model_device=lambda: torch.device('cpu'),
        _hmm_dtype=lambda: torch.float32, _prepare_gene_hmm_inputs=lambda p, d: torch.as_tensor(p, dtype=torch.float32),
        sequence_prediction=lambda *a, **kw: pytest.fail('Must reuse provided probabilities'))


def test_joint_decode_reuses_probabilities_and_keeps_whole_intron_memory():
    # STOP -> 3UTR/intron -> 5UTR -> START. Retain both existing CDS ends,
    # resolving the incompatible intron by a scored IR separator, not an ORF.
    path = np.array([14, 16, 17, 18, 18, 19, 15, 7], dtype=np.int32)
    protected = np.array([True, False, False, False, False, False, False, True])
    desired = np.array([14, 16, 16, 0, 0, 15, 15, 7])
    probabilities = np.full((len(path), 20), .001, dtype=np.float32)
    probabilities[np.arange(len(path)), desired] = .9
    nuc = np.eye(5, dtype=np.float32)[np.full(len(path), 4)]
    fixed, details = joint_decode(toy_predictor(), nuc, path, protected, 0, probabilities)
    np.testing.assert_array_equal(fixed, desired)
    assert details['probability_source'] == 'retained_probabilities'
    assert details['joint_decode']


def test_noncoding_competitor_explains_conflict_without_forcing_an_orf():
    path = np.array([0, 15, 17, 18, 19, 16, 0], dtype=np.int32)
    probabilities = np.full((len(path), 20), .01, dtype=np.float32)
    probabilities[np.arange(len(path)), path] = .99
    protected = np.array([True, False, False, False, False, False, True])
    nuc = np.eye(5, dtype=np.float32)[np.full(len(path), 4)]
    fixed, _ = joint_decode(toy_predictor(), nuc, path, protected, 0, probabilities)
    np.testing.assert_array_equal(fixed, np.zeros(len(path)))


def test_unsatisfiable_emissions_report_failure():
    path = np.array([0, 15, 17, 18, 19, 16, 0], dtype=np.int32)
    probabilities = np.zeros((len(path), 20), dtype=np.float32)
    protected = np.array([True, False, False, False, False, False, True])
    nuc = np.eye(5, dtype=np.float32)[np.full(len(path), 4)]
    with pytest.raises(ValueError, match='No legal'):
        joint_decode(toy_predictor(), nuc, path, protected, 0, probabilities)


def test_joint_decode_can_select_a_scored_coding_alternative():
    # A coding result is allowed when it wins the likelihood competition; it
    # must not be produced by relabeling UTRs or forbidden by a hard IR mask.
    path = np.array([0, 15, 17, 18, 18, 18, 19, 16, 0], dtype=np.int32)
    desired = np.array([0, 7, 5, 6, 4, 5, 14, 0, 0])
    probabilities = np.full((len(path), 20), .0001, dtype=np.float32)
    probabilities[np.arange(len(path)), desired] = .99
    protected = np.array([True] + [False]*7 + [True])
    nuc = np.eye(5, dtype=np.float32)[np.full(len(path), 4)]
    fixed, _ = joint_decode(toy_predictor(), nuc, path, protected, 0, probabilities)
    np.testing.assert_array_equal(fixed, desired)
