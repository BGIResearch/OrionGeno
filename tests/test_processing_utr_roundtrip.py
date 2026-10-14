"""Replay actual processing truth through production assembly and public GFF.

The portable fixture was made by the pinned processing branch's real H5 writer.
Legacy GFF was captured with the original main branch's real assembly methods.
These tests exercise no neural checkpoint or HMM; labels are perfect truth, so
an exon/phase mismatch belongs to the state-to-annotation contract itself.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


FIXTURE_PATH = Path(__file__).parent / "fixtures/processing_utr_roundtrip.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
CASES = FIXTURE["cases"]
LAYOUTS = [(case, width) for case in CASES for width in case["window_sizes"]]
LAYOUT_IDS = [f"{case['id']}-window{width}" for case, width in LAYOUTS]
PROCESSING_COMMIT = "73d7225543e578d0f76a35269cc384ae13878f7d"


def _sha256(value):
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def _rows(gff3):
    return [line.split("\t") for line in gff3.splitlines() if line and not line.startswith("#")]


def _structure(gff3):
    rows = _rows(gff3)
    exons = sorted([int(row[3]), int(row[4])] for row in rows if row[2] == "exon")
    return {
        "exons": exons,
        "introns": [[left[1] + 1, right[0] - 1] for left, right in zip(exons, exons[1:])],
        "cds_with_phase": sorted([int(row[3]), int(row[4]), int(row[7])]
                                 for row in rows if row[2] == "CDS"),
        "transcript_count": sum(row[2] in {"mRNA", "transcript"} for row in rows),
        "gene_count": sum(row[2] == "gene" for row in rows),
    }


def test_portable_processing_fixture_provenance_and_original_gold_agree():
    assert FIXTURE["schema_version"] == 1
    assert FIXTURE["provenance"]["processing_commit"] == PROCESSING_COMMIT
    assert FIXTURE["provenance"]["legacy_main_commit"] == "90442a2899172c7bdd2bbda19d81c8853082cdb4"
    payload = {key: value for key, value in FIXTURE.items() if key != "fixture_payload_sha256"}
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert _sha256(serialized) == FIXTURE["fixture_payload_sha256"]
    assert {case["group"] for case in CASES} == {
        "plus_5utr", "plus_3utr", "minus_5utr", "minus_3utr",
        "multiple_utr_introns", "cds_and_utr", "contig_window_edge",
    }
    for case in CASES:
        assert _sha256(case["source_fasta"]) == case["source_sha256"]["fasta"]
        assert _sha256(case["source_gff3"]) == case["source_sha256"]["gff3"]
        assert _sha256(case["processing_gold_gff3"]) == case["processing_gold_sha256"]
        assert _structure(case["source_gff3"]) == _structure(case["processing_gold_gff3"])
        assert _structure(case["processing_gold_gff3"]) == case["corrected_expected"]
        states = np.asarray(case["oriented_state_ids"], dtype=np.uint8)
        assert _sha256(states.tobytes()) == case["oriented_state_ids_sha256"]
        assert len(states) == len(case["sequence"]) == len(case["oriented_sequence"])
        assert states.min() >= 0 and states.max() < 20
        assert 17 in states and 19 in states
        assert case["h5_contract"]["gene_label_dim"] == 20
        assert case["h5_contract"]["aux_channel_count"] == 9
        for legacy in case["legacy_outputs"]:
            assert _sha256(legacy["gff3"]) == legacy["gff3_sha256"]
            assert _structure(legacy["gff3"]) == legacy["structure"]
            assert legacy["structure"]["cds_with_phase"] == case["corrected_expected"]["cds_with_phase"]
            assert legacy["structure"]["exons"] != case["corrected_expected"]["exons"]
            assert legacy["structure"]["introns"] != case["corrected_expected"]["introns"]
    # The CDS+UTR group deliberately includes phases 1 and 2, not only phase 0.
    assert {phase for case in CASES if case["group"] == "cds_and_utr"
            for _start, _end, phase in case["corrected_expected"]["cds_with_phase"]} == {0, 1, 2}


@pytest.mark.integration
@pytest.mark.parametrize("case,width", LAYOUTS, ids=LAYOUT_IDS)
def test_processing_truth_to_transcript_and_public_gff(case, width, tmp_path):
    pytest.importorskip("torch")
    from oriongeno.prediction_annotator import PredictionAnnotator
    from oriongeno.prediction_format import write_prediction_annotation

    length = len(case["sequence"])
    labels = np.asarray(case["oriented_state_ids"], dtype=np.uint8).reshape(-1, width).copy()
    coords = []
    for start in range(0, length, width):
        left, right = start + 1, start + width
        if case["strand"] == "-":
            left, right = length - start - width + 1, length - start
        coords.append([case["chrom"], case["strand"], left, right])

    # Constructor does not load a checkpoint. Truth bypasses only NN and HMM;
    # coordinate reversal, stitch, transcript grouping and phase are production.
    predictor = PredictionAnnotator(batch_size=1)
    annotation, transcript_number = predictor.create_gene_annotation(
        labels, coords, np.zeros((*labels.shape, 5), dtype=np.uint8),
        strand=case["strand"], filt=False, allow_boundary_reprediction=False,
    )
    annotation.norm_tx_format()
    annotation.find_genes()
    assert transcript_number == 1
    assert len(annotation.transcripts) == 1
    transcript = next(iter(annotation.transcripts.values()))
    assert transcript.strand == case["strand"]
    assert transcript.chr == case["chrom"]
    assert sorted(transcript.get_type_coords("exon", False)) == case["corrected_expected"]["exons"]
    assert sorted(transcript.get_type_coords("intron", False)) == case["corrected_expected"]["introns"]
    assert sorted([int(row[3]), int(row[4]), int(row[7])]
                  for row in transcript.transcript_lines["CDS"]) == case["corrected_expected"]["cds_with_phase"]
    for internal_feature, gold_feature in (("5UTR", "five_prime_UTR"), ("3UTR", "three_prime_UTR")):
        expected = sorted([int(row[3]), int(row[4])]
                          for row in _rows(case["processing_gold_gff3"]) if row[2] == gold_feature)
        assert sorted(transcript.get_type_coords(internal_feature, False)) == expected

    output = tmp_path / f"{case['id']}.gff3"
    write_prediction_annotation(annotation, str(output), genome={case["chrom"]: case["sequence"]})
    gff3 = output.read_text(encoding="utf-8")
    assert _structure(gff3) == case["corrected_expected"]
    for row in _rows(gff3):
        assert row[0] == case["chrom"] and row[6] == case["strand"]
        assert row[2] in {"gene", "mRNA", "exon", "CDS"}
    mrna = next(row for row in _rows(gff3) if row[2] == "mRNA")
    assert "StartCodon=ATG" in mrna[8] and "StopCodon=TAA" in mrna[8]
    legacy = next(item for item in case["legacy_outputs"] if item["window_size"] == width)
    assert _structure(gff3)["exons"] != legacy["structure"]["exons"]
    assert _structure(gff3)["cds_with_phase"] == legacy["structure"]["cds_with_phase"]
    assert all(check["status"] == "joined" for check in predictor.boundary_checks)


@pytest.mark.integration
def test_optional_real_processing_regeneration():
    processing_root = os.environ.get("ORIONGENO_PROCESSING_ROOT")
    if not processing_root:
        pytest.skip("Set ORIONGENO_PROCESSING_ROOT to verify fixture against the pinned real processing checkout")
    script = FIXTURE_PATH.with_name("regenerate_processing_utr_fixtures.py")
    result = subprocess.run(
        [sys.executable, str(script), "--processing-root", processing_root, "--check"],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
