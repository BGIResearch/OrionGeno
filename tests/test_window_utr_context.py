"""Regressions for Danio's cross-window 5UTR/3UTR intron conflict."""
import numpy as np
import pytest

# Importing the HMM package requires torch; lightweight CI omits inference dependencies.
pytest.importorskip("torch")

from oriongeno.data_processing.state_features import gene_state_ranges, resolve_utr_boundary_ranges  # noqa: E402
from oriongeno.data_processing.window_stitching import stitch_window_ranges, _conflicting_utr_context  # noqa: E402


def _stitch(rows, strand="+", start=1):
    labels = [np.asarray(row, dtype=np.uint8) for row in rows]
    coords = []
    for row in labels:
        coords.append(["chr", strand, start, start + len(row) - 1])
        start += len(row)
    blocks, checks = stitch_window_ranges(labels, coords, strand, gene_state_ranges)
    return labels, coords, blocks, checks


@pytest.mark.parametrize("strand", ["+", "-"])
@pytest.mark.parametrize("middle_windows", [0, 1, 3])
def test_opposite_utr_sides_split_across_intron_windows(strand, middle_windows):
    rows = [[0, 15, 17, 18]] + [[18] * 4 for _ in range(middle_windows)] + [[18, 19, 16, 0]]
    if strand == "-":
        rows = [row[::-1] for row in rows[::-1]]
    _, _, blocks, checks = _stitch(rows, strand)
    assert len(blocks) == 2
    assert checks[-1]["status"] == "utr_context_conflict"
    assert all(c["status"] == "joined" for c in checks[:-1])
    for _, ranges in blocks:
        resolve_utr_boundary_ranges(ranges, allow_unresolved=True)


@pytest.mark.parametrize("strand", ["+", "-"])
@pytest.mark.parametrize("side", [15, 16])
def test_matching_utr_context_still_joins_and_resolves(strand, side):
    rows = [[side, 17, 18], [18, 18], [18, 19, side]]
    if strand == "-":
        rows = [row[::-1] for row in rows[::-1]]
    _, _, blocks, checks = _stitch(rows, strand)
    assert len(blocks) == 1
    assert all(c["status"] == "joined" for c in checks)
    assert resolve_utr_boundary_ranges(blocks[0][1]) == [
        ["5UTR" if side == 15 else "3UTR", 1, 2], ["intron", 3, 6],
        ["5UTR" if side == 15 else "3UTR", 7, 8],
    ]


@pytest.mark.parametrize("barrier", ["CDS", "intergenic"])
def test_context_does_not_cross_cds_or_intergenic(barrier):
    assert not _conflicting_utr_context(
        [["5UTR", 1, 2], [barrier, 3, 4], ["3UTR", 5, 6], ["intron", 7, 8]],
        [["intron", 9, 10], ["3UTR", 11, 12]],
    )
    assert not _conflicting_utr_context(
        [["5UTR", 1, 2], ["intron", 3, 4]],
        [["intron", 5, 6], ["5UTR", 7, 8], [barrier, 9, 10], ["3UTR", 11, 12]],
    )


def test_coordinate_gap_is_not_utr_context():
    assert not _conflicting_utr_context([["5UTR", 1, 2], ["intron", 3, 4]],
                                        [["intron", 6, 7], ["3UTR", 8, 9]])


def test_strict_resolver_still_rejects_conflicting_context():
    with pytest.raises(ValueError, match="Conflicting"):
        resolve_utr_boundary_ranges([["5UTR", 1, 2], ["intron", 3, 4], ["3UTR", 5, 6]])


def test_danio_genomic_coordinates_reject_the_observed_join():
    # Ranges captured by local inference, including exon-side splice bases.
    left_start, boundary, end = 10880001, 11520000, 12160000
    left = np.zeros(boundary - left_start + 1, dtype=np.uint8)
    right = np.zeros(end - boundary, dtype=np.uint8)
    for start, stop, state in [(11392866, 11392866, 15), (11392867, 11392867, 17),
                              (11392868, 11436982, 18), (11436983, 11436983, 19),
                              (11436984, 11437019, 15), (11437020, 11437020, 17),
                              (11437021, boundary, 18)]:
        left[start-left_start:stop-left_start+1] = state
    right[:11531728-boundary] = 18
    right[11531729-boundary-1] = 19
    right[11531730-boundary-1] = 16
    _, _, blocks, checks = _stitch([left, right], start=left_start)
    assert len(blocks) == 2
    assert checks[0]["left_end"] == 11520000
    assert checks[0]["right_start"] == 11520001
    assert checks[0]["status"] == "utr_context_conflict"


@pytest.mark.integration
@pytest.mark.parametrize("strand", ["+", "-"])
def test_conflict_does_not_crash_assembly_or_remove_unrelated_cds(strand):
    from oriongeno.prediction_annotator import PredictionAnnotator
    rows = [[0, 15, 17, 18], [18, 19, 16, 0, 7, 5, 6, 4, 5, 14, 0]]
    length = sum(map(len, rows))
    labels = [np.asarray(row, dtype=np.uint8) for row in rows]
    coords, start = [], 1
    for row in labels:
        end = start + len(row) - 1
        coords.append(["chr", strand, start if strand == "+" else length-end+1,
                       end if strand == "+" else length-start+1])
        start = end + 1
    predictor = PredictionAnnotator(batch_size=1)
    annotation, count = predictor.create_gene_annotation(
        labels, coords, [np.zeros((len(row), 5), dtype=np.uint8) for row in rows],
        strand=strand, filt=False, allow_boundary_reprediction=False,
    )
    assert count == len(annotation.transcripts) == 1
    tx = next(iter(annotation.transcripts.values()))
    assert tx.get_cds_len() == 6
    assert predictor.boundary_checks[0]["status"] == "utr_context_conflict"
