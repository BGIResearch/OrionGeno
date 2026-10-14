import itertools

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# These modules require torch, so import them only after the optional-dependency guard.
from oriongeno.hmm_layer import GenePredHMMLayer  # noqa: E402
from oriongeno.hmm_layer.coding_constraints import (  # noqa: E402
    _graph,
    _legal,
    _decode,
    _competition_graph,
    _short_orf_competition_regions,
)


STOPS = ('TAA', 'TAG', 'TGA')
CASES = [
    ('ATGCCCTAA' + 'GTAAAAAG' + 'CCCTAG',
     [7,5,6,4,5,6,4,5,9] + [2]*8 + [12,5,6,4,5,14]),
    ('ATGCCCT' + 'GTAAAAAG' + 'AACCCTAG',
     [7,5,6,4,5,6,10] + [3]*8 + [13,6,4,5,6,4,5,14]),
    ('ATGCCCTA' + 'GTAAAAAG' + 'ACCCTAG',
     [7,5,6,4,5,6,4,8] + [1]*8 + [11,4,5,6,4,5,14]),
    ('ATGCCCTAG' + 'CGTAC' + 'ATGCCCTAG',
     [7,5,6,4,5,6,4,5,14] + [16,17,18,19,15] + [7,5,6,4,5,6,4,5,14]),
]


def bases(sequence):
    return np.array(['ACGTN'.index(c) for c in sequence], dtype=np.int32)


@pytest.mark.parametrize('sequence,path', CASES)
def test_illegal_optimum_is_redecoded_not_filtered(sequence, path):
    stops, labels, initial, sources, targets, allowed = _graph(1, STOPS)
    nuc = bases(sequence)
    path = np.array(path, dtype=np.int32)
    assert not _legal(path, nuc, stops, allowed)
    emit = np.full((len(path), 20), 1e-4, dtype=np.float32)
    emit[np.arange(len(path)), path] = 1
    repaired = _decode(emit, np.zeros((20,20), dtype=np.float32),
                       np.ones(20, dtype=np.float32), nuc, labels, initial, sources, targets)
    assert len(repaired) == len(path)
    assert _legal(repaired, nuc, stops, allowed)
    assert not np.array_equal(repaired, path)


@pytest.mark.parametrize('sequence,path', CASES[:3])
def test_unknown_codon_is_not_fabricated_as_stop(sequence, path):
    stops, *_, allowed = _graph(1, STOPS)
    assert _legal(np.array(path), bases('N'*len(sequence)), stops, allowed)


@pytest.mark.parametrize('side', [15,16])
def test_utr_introns_preserve_each_side(side):
    stops, *_, allowed = _graph(1, STOPS)
    path = np.array([side,17,18,18,19,side])
    assert _legal(path, bases('CCCCCC'), stops, allowed)
    path[-1] = 31-side
    assert not _legal(path, bases('CCCCCC'), stops, allowed)


def test_donor_third_base_has_zero_stop_emission():
    hmm = GenePredHMMLayer()
    hmm.cell.recurrent_init()
    sequence, path = CASES[0]
    nuc = torch.nn.functional.one_hot(torch.tensor(np.array([bases(sequence)]), dtype=torch.long),5).float()
    probs = torch.full((1,len(sequence),20), .05)
    emission = hmm.cell.emission_probs(hmm.concat_inputs(probs,nuc), training=False)
    assert emission[0,0,8,9] == 0


def test_old_checkpoint_cannot_restore_missing_donor_mask():
    hmm = GenePredHMMLayer()
    saved = {k: v.clone() for k,v in hmm.state_dict().items()}
    keys = [k for k in saved if k.endswith('codon_probs')]
    assert keys
    for key in keys:
        saved[key].fill_(1/64)
    hmm.load_state_dict(saved)
    restored = hmm.state_dict()
    for key in keys:
        assert not torch.equal(restored[key], saved[key])


def test_public_viterbi_repairs_split_stop(monkeypatch):
    from oriongeno.hmm_layer import Viterbi
    sequence, path = CASES[1]
    monkeypatch.setattr(Viterbi, 'viterbi', lambda *a, **k: torch.tensor([[path]]))
    hmm = GenePredHMMLayer()
    nuc = torch.nn.functional.one_hot(torch.tensor(np.array([bases(sequence)]), dtype=torch.long),5).float()
    probs = torch.full((1,len(sequence),20), .0001)
    probs[0,torch.arange(len(path)),torch.tensor(path)] = .99
    with torch.no_grad():
        result = hmm.viterbi(probs,nuc)
    stops, *_, allowed = _graph(1, STOPS)
    assert _legal(result[0].numpy(), bases(sequence), stops, allowed)


def test_repair_honors_endpoint_hints_and_multiple_batch_items(monkeypatch):
    from oriongeno.hmm_layer import Viterbi
    sequence, path = CASES[1]
    monkeypatch.setattr(Viterbi, 'viterbi', lambda *a, **k: torch.tensor([[path,path]]))
    hmm = GenePredHMMLayer()
    nuc = torch.nn.functional.one_hot(torch.tensor(np.array([bases(sequence)]*2), dtype=torch.long),5).float()
    probs = torch.full((2,len(sequence),20), .05)
    hints = torch.zeros((2,2,20))
    hints[0,0,7] = hints[0,1,14] = 1
    hints[1,:,0] = 1
    with torch.no_grad():
        result = hmm.viterbi(probs,nuc,end_hints=hints)
    assert result[0,0] == 7 and result[0,-1] == 14
    assert result[1,0] == 0 and result[1,-1] == 0


def test_configured_stop_codons_and_multiple_copies():
    sequence, path = CASES[1]
    path = np.array(path) + 19
    stops, *_, allowed = _graph(2, ('TAG','TGA'))
    assert _legal(path,bases(sequence),stops,allowed)  # TAA is not a stop in this configuration.
    stops, *_, allowed = _graph(2, STOPS)
    assert not _legal(path,bases(sequence),stops,allowed)


def test_constrained_score_matches_exhaustive_short_paths():
    stops, labels, initial, sources, targets, allowed = _graph(1, STOPS)
    rng = np.random.default_rng(42)
    emit = rng.uniform(.01,1,(3,20)).astype(np.float32)
    transition = rng.uniform(-2,0,(20,20)).astype(np.float32)
    init = np.ones(20,dtype=np.float32)
    nuc = bases('TAA')
    decoded = _decode(emit,transition,init,nuc,labels,initial,sources,targets)
    def score(path):
        return sum(np.log(emit[i,s]) for i,s in enumerate(path)) + sum(transition[a,b] for a,b in zip(path,path[1:]))
    optimum = max(score(p) for p in itertools.product(range(20),repeat=3)
                  if _legal(np.array(p),nuc,stops,allowed))
    assert score(decoded) == pytest.approx(optimum, abs=1e-5)


def test_noncoding_signal_can_win_without_inventing_a_cds():
    path = np.array([0,15,17,18,18,19,16,0])
    emit = np.full((len(path),20),1e-6,dtype=np.float32)
    emit[np.arange(len(path)),path] = 1
    g = _competition_graph(1,STOPS)
    decoded = _decode(emit,np.zeros((20,20),dtype=np.float32),
                      np.ones(20,dtype=np.float32),bases('CCCCCCCC'),
                      *g[1:5],g[6])
    assert np.all(decoded == 0)


def test_noncoding_branch_does_not_mask_supported_short_coding_path():
    path = np.array([0,7,5,6,4,5,14,0])
    emit = np.full((len(path),20),1e-6,dtype=np.float32)
    emit[np.arange(len(path)),path] = 1
    g = _competition_graph(1,STOPS)
    decoded = _decode(emit,np.zeros((20,20),dtype=np.float32),
                      np.ones(20,dtype=np.float32),bases('CATGTGAC'),
                      *g[1:5],g[6])
    assert np.array_equal(decoded,path)


def test_public_noncoding_competition_keeps_utr_endpoint_hints(monkeypatch):
    from oriongeno.hmm_layer import Viterbi
    path=[15,17]+[18]*10+[19,16]
    monkeypatch.setattr(Viterbi,'viterbi',lambda *a,**k:torch.tensor([[path]]))
    hmm=GenePredHMMLayer()
    nuc=torch.nn.functional.one_hot(torch.tensor([['ACGT'.index(c) for c in 'CATGCCCTAGCCCC']]),5).float()
    probs=torch.full((1,14,20),.05)
    hints=torch.zeros((1,2,20))
    hints[0,0,15]=hints[0,1,16]=1
    with torch.no_grad():
        result=hmm.viterbi(probs,nuc,end_hints=hints)
    assert result[0,0]==15 and result[0,-1]==16


def test_noncoding_competition_protects_long_and_partial_models():
    short=[7,5,6,4,5,14]
    long=[7,5,6]+[4,5,6]*18+[4,5,14]
    partial=[4,5,6]*4
    path=np.array([0]+short+[0]+long+[0]+partial)
    eligible,protected=_short_orf_competition_regions(path)
    assert eligible
    assert not protected[1:7].any()
    assert protected[8:68].all()
    assert protected[-12:].all()
    assert not _short_orf_competition_regions(np.array(long))[0]


def test_noncoding_decode_cannot_displace_anchored_coding_states():
    path=np.array([0,7,5,6,4,5,14,0])
    g=_competition_graph(1,STOPS)
    # Deliberately favor noncoding labels; coding anchors must still survive.
    emit=np.full((8,20),.01,dtype=np.float32)
    emit[:,15:]=1
    protected=np.array([False]+[True]*6+[False])
    decoded=_decode(emit,np.zeros((20,20),dtype=np.float32),np.ones(20,dtype=np.float32),
                    bases('CATGTGAC'),*g[1:5],g[6],None,g[7],protected,path)
    assert np.array_equal(decoded[1:7],path[1:7])
