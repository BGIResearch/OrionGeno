#!/usr/bin/env python3
"""Evaluate predicted genes by exact CDS/ORF structure.

The metric intentionally ignores UTRs, exon features, and transcript/gene
outer boundaries. A reference gene may have multiple transcript ORFs. A
predicted gene is counted as correct when one of its CDS-only transcript ORFs
matches any transcript ORF from one unmatched reference gene.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Tuple


Interval = Tuple[str, str, int, int]
OrfKey = Tuple[Interval, ...]

TRANSCRIPT_FEATURES = {
    "transcript",
    "mrna",
    "rna",
    "primary_transcript",
    "lnc_rna",
    "ncrna",
}


@dataclass(frozen=True)
class TranscriptOrf:
    gene_id: str
    transcript_id: str
    orf_key: OrfKey

    @property
    def chrom(self) -> str:
        return self.orf_key[0][0] if self.orf_key else ""

    @property
    def strand(self) -> str:
        return self.orf_key[0][1] if self.orf_key else ""

    @property
    def cds_exon_count(self) -> int:
        return len(self.orf_key)

    @property
    def cds_span(self) -> str:
        if not self.orf_key:
            return ""
        starts = [interval[2] for interval in self.orf_key]
        ends = [interval[3] for interval in self.orf_key]
        return f"{min(starts)}-{max(ends)}"

    @property
    def cds_intervals(self) -> str:
        return ",".join(
            f"{chrom}:{strand}:{start}-{end}"
            for chrom, strand, start, end in self.orf_key
        )


@dataclass
class ParsedAnnotation:
    transcripts: Dict[str, TranscriptOrf]
    genes: Dict[str, List[TranscriptOrf]]
    raw_transcript_count: int
    cds_feature_count: int
    orphan_transcript_count: int


@dataclass
class MatchResult:
    reference: ParsedAnnotation
    prediction: ParsedAnnotation
    matched_pairs: List[Tuple[TranscriptOrf, TranscriptOrf]]
    unmatched_prediction_genes: List[str]
    unmatched_reference_genes: List[str]
    prediction_genes_with_multiple_orfs: List[str]

    @property
    def matched_gene_count(self) -> int:
        return len(self.matched_pairs)

    @property
    def precision(self) -> float:
        total = len(self.prediction.genes)
        return self.matched_gene_count / total if total else 0.0

    @property
    def recall(self) -> float:
        total = len(self.reference.genes)
        return self.matched_gene_count / total if total else 0.0


def parse_attributes(attribute_text: str) -> Dict[str, List[str]]:
    """Parse GTF or GFF3 attributes into a key -> values mapping."""
    attrs: Dict[str, List[str]] = defaultdict(list)
    for item in attribute_text.strip().rstrip(";").split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
        elif " " in item:
            key, value = item.split(None, 1)
        else:
            attrs[item].append("")
            continue
        key = key.strip()
        value = value.strip().strip('"')
        for part in value.split(","):
            part = part.strip().strip('"')
            if part:
                attrs[key].append(part)
    return attrs


def first_attr(attrs: Dict[str, List[str]], *keys: str) -> Optional[str]:
    for key in keys:
        values = attrs.get(key)
        if values:
            return values[0]
    return None


def attr_values(attrs: Dict[str, List[str]], *keys: str) -> List[str]:
    values: List[str] = []
    for key in keys:
        values.extend(attrs.get(key, []))
    return values


def infer_transcript_ids(feature_type: str, attrs: Dict[str, List[str]]) -> List[str]:
    transcript_id = first_attr(attrs, "transcript_id", "transcriptId")
    if transcript_id:
        return [transcript_id]
    if feature_type.lower() == "cds":
        parents = attr_values(attrs, "Parent", "parent")
        if parents:
            return parents
    feature_id = first_attr(attrs, "ID", "id")
    return [feature_id] if feature_id else []


def infer_gene_id_for_feature(feature_type: str, attrs: Dict[str, List[str]]) -> Optional[str]:
    gene_id = first_attr(attrs, "gene_id", "geneID", "gene", "gene_name")
    if gene_id:
        return gene_id
    if feature_type.lower() in TRANSCRIPT_FEATURES:
        parent = first_attr(attrs, "Parent", "parent")
        if parent:
            return parent
    if feature_type.lower() == "gene":
        return first_attr(attrs, "ID", "id")
    return None


def read_annotation(path: str) -> ParsedAnnotation:
    """Read CDS-only transcript ORFs grouped by gene."""
    transcript_to_gene: Dict[str, str] = {}
    transcript_cds: Dict[str, List[Interval]] = defaultdict(list)
    raw_transcripts = set()
    cds_feature_count = 0

    with open(path, "r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue

            chrom, _, feature_type, start_text, end_text, _, strand, _, attr_text = fields[:9]
            feature_key = feature_type.lower()
            attrs = parse_attributes(attr_text)

            if feature_key == "gene":
                continue

            gene_id = infer_gene_id_for_feature(feature_type, attrs)
            transcript_ids = infer_transcript_ids(feature_type, attrs)

            if feature_key in TRANSCRIPT_FEATURES:
                for transcript_id in transcript_ids:
                    raw_transcripts.add(transcript_id)
                    if gene_id:
                        transcript_to_gene.setdefault(transcript_id, gene_id)
                continue

            if feature_key != "cds":
                continue

            try:
                start = int(start_text)
                end = int(end_text)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid CDS coordinates") from error
            if end < start:
                start, end = end, start

            cds_feature_count += 1
            if not transcript_ids:
                transcript_ids = [f"__orphan_cds_line_{line_number}"]

            for transcript_id in transcript_ids:
                raw_transcripts.add(transcript_id)
                if gene_id:
                    transcript_to_gene.setdefault(transcript_id, gene_id)
                transcript_cds[transcript_id].append((chrom, strand, start, end))

    transcripts: Dict[str, TranscriptOrf] = {}
    genes: Dict[str, List[TranscriptOrf]] = defaultdict(list)
    orphan_count = 0

    for transcript_id, intervals in transcript_cds.items():
        if not intervals:
            continue
        gene_id = transcript_to_gene.get(transcript_id)
        if not gene_id:
            gene_id = transcript_id
            orphan_count += 1
        orf_key = tuple(sorted(intervals, key=lambda item: (item[0], item[1], item[2], item[3])))
        transcript = TranscriptOrf(
            gene_id=gene_id,
            transcript_id=transcript_id,
            orf_key=orf_key,
        )
        transcripts[transcript_id] = transcript
        genes[gene_id].append(transcript)

    return ParsedAnnotation(
        transcripts=transcripts,
        genes=dict(genes),
        raw_transcript_count=len(raw_transcripts),
        cds_feature_count=cds_feature_count,
        orphan_transcript_count=orphan_count,
    )


def choose_prediction_gene_match(
    prediction_orfs: Iterable[TranscriptOrf],
    reference_index: Dict[OrfKey, Deque[TranscriptOrf]],
) -> Optional[Tuple[TranscriptOrf, TranscriptOrf]]:
    """Return one prediction/reference transcript pair for a prediction gene."""
    for predicted_orf in prediction_orfs:
        candidates = reference_index.get(predicted_orf.orf_key)
        if candidates:
            reference_orf = candidates.popleft()
            return predicted_orf, reference_orf
    return None


def calculate_matches(reference: ParsedAnnotation, prediction: ParsedAnnotation) -> MatchResult:
    """Match prediction genes against reference genes by exact ORF equality."""
    reference_index: Dict[OrfKey, Deque[TranscriptOrf]] = defaultdict(deque)
    for gene_id in sorted(reference.genes):
        # A reference gene may have multiple transcript ORFs. All are valid
        # targets, but the gene receives credit at most once.
        unique_orfs = {}
        for transcript in reference.genes[gene_id]:
            unique_orfs.setdefault(transcript.orf_key, transcript)
        for transcript in unique_orfs.values():
            reference_index[transcript.orf_key].append(transcript)

    matched_pairs: List[Tuple[TranscriptOrf, TranscriptOrf]] = []
    matched_reference_genes = set()
    matched_prediction_genes = set()
    prediction_genes_with_multiple_orfs: List[str] = []

    for gene_id in sorted(prediction.genes):
        unique_prediction_orfs = {}
        for transcript in prediction.genes[gene_id]:
            unique_prediction_orfs.setdefault(transcript.orf_key, transcript)
        if len(unique_prediction_orfs) > 1:
            prediction_genes_with_multiple_orfs.append(gene_id)

        match = choose_prediction_gene_match(unique_prediction_orfs.values(), reference_index)
        if match is None:
            continue
        predicted_orf, reference_orf = match
        matched_pairs.append((predicted_orf, reference_orf))
        matched_prediction_genes.add(predicted_orf.gene_id)
        matched_reference_genes.add(reference_orf.gene_id)

        # Remove all remaining transcript ORFs from the matched reference gene,
        # so duplicate predictions cannot score multiple hits on the same gene.
        for reference_transcript in reference.genes[reference_orf.gene_id]:
            queue = reference_index.get(reference_transcript.orf_key)
            if not queue:
                continue
            reference_index[reference_transcript.orf_key] = deque(
                item for item in queue if item.gene_id != reference_orf.gene_id
            )

    unmatched_prediction_genes = sorted(set(prediction.genes) - matched_prediction_genes)
    unmatched_reference_genes = sorted(set(reference.genes) - matched_reference_genes)

    return MatchResult(
        reference=reference,
        prediction=prediction,
        matched_pairs=matched_pairs,
        unmatched_prediction_genes=unmatched_prediction_genes,
        unmatched_reference_genes=unmatched_reference_genes,
        prediction_genes_with_multiple_orfs=sorted(prediction_genes_with_multiple_orfs),
    )


def print_logic() -> None:
    print("Matching logic:")
    print("  1. Only CDS features are used to define an ORF.")
    print("  2. UTR, exon, gene boundary, and transcript boundary coordinates are ignored.")
    print("  3. One transcript ORF is the exact set of CDS intervals: chrom, strand, start, end.")
    print("  4. A reference gene may contain multiple transcript ORFs; any one may match.")
    print("  5. A predicted gene is counted once if one predicted ORF matches one reference transcript ORF.")
    print("  6. Each reference gene can be matched once, so duplicate predictions do not get extra credit.")
    print("  7. Gene and transcript IDs do not need to be the same; they are used for grouping/reporting only.")


def print_summary(result: MatchResult) -> None:
    print("\nInput summary:")
    print(f"  Reference CDS-bearing genes: {len(result.reference.genes)}")
    print(f"  Reference CDS-bearing transcripts: {len(result.reference.transcripts)}")
    print(f"  Prediction CDS-bearing genes: {len(result.prediction.genes)}")
    print(f"  Prediction CDS-bearing transcripts: {len(result.prediction.transcripts)}")
    print(f"  Reference CDS features read: {result.reference.cds_feature_count}")
    print(f"  Prediction CDS features read: {result.prediction.cds_feature_count}")
    if result.reference.orphan_transcript_count or result.prediction.orphan_transcript_count:
        print(
            "  CDS transcripts without gene IDs: "
            f"reference={result.reference.orphan_transcript_count}, "
            f"prediction={result.prediction.orphan_transcript_count}"
        )
    if result.prediction_genes_with_multiple_orfs:
        print(
            "  Prediction genes with multiple distinct ORFs: "
            f"{len(result.prediction_genes_with_multiple_orfs)}"
        )

    print("\nGene-level ORF metrics:")
    print(f"  Matched prediction genes: {result.matched_gene_count}")
    print(f"  Precision: {result.matched_gene_count} / {len(result.prediction.genes)} = {result.precision:.4f}")
    print(f"  Recall: {result.matched_gene_count} / {len(result.reference.genes)} = {result.recall:.4f}")


def write_matches(path: str, result: MatchResult) -> None:
    with open(path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            [
                "prediction_gene_id",
                "prediction_transcript_id",
                "reference_gene_id",
                "reference_transcript_id",
                "chrom",
                "strand",
                "cds_exon_count",
                "cds_span",
                "cds_intervals",
            ]
        )
        for predicted, reference in result.matched_pairs:
            writer.writerow(
                [
                    predicted.gene_id,
                    predicted.transcript_id,
                    reference.gene_id,
                    reference.transcript_id,
                    predicted.chrom,
                    predicted.strand,
                    predicted.cds_exon_count,
                    predicted.cds_span,
                    predicted.cds_intervals,
                ]
            )


def write_unmatched(path: str, result: MatchResult) -> None:
    with open(path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["side", "gene_id"])
        for gene_id in result.unmatched_prediction_genes:
            writer.writerow(["prediction", gene_id])
        for gene_id in result.unmatched_reference_genes:
            writer.writerow(["reference", gene_id])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage=(
            "%(prog)s reference.gtf prediction.gtf [matched_orf_pairs.csv] "
            "[--unmatched-output unmatched_genes.csv]"
        ),
        description=(
            "Compare a reference annotation and a prediction by exact CDS/ORF "
            "structure. UTRs and transcript/gene boundary differences are ignored."
        ),
        epilog=(
            "Example: python tests/gtf_multi_trans_acc.py "
            "tests/Arabidopsis_thaliana/Arabidopsis_thaliana.gold.gtf "
            "tests/Arabidopsis_thaliana/Arabidopsis_thaliana_oriongeno.gtf "
            "matched_orf_pairs.csv"
        ),
    )
    parser.add_argument("reference", metavar="reference.gtf", help="Reference GTF/GFF3 annotation file")
    parser.add_argument("prediction", metavar="prediction.gtf", help="Predicted GTF/GFF3 annotation file")
    parser.add_argument(
        "matched_output",
        metavar="matched_orf_pairs.csv",
        nargs="?",
        help="Optional CSV path for matched prediction/reference ORF pairs.",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="matched_orf_pairs.csv",
        help="Optional CSV path for matched prediction/reference ORF pairs.",
    )
    parser.add_argument(
        "--unmatched-output",
        help="Optional CSV path for unmatched prediction and reference genes.",
    )
    parser.add_argument(
        "--quiet-logic",
        action="store_true",
        help="Do not print the matching-rule explanation.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.output and args.matched_output:
        parser.error("Use either the third positional output path or -o/--output, not both.")
    matched_output = args.output or args.matched_output

    if not args.quiet_logic:
        print_logic()

    print(f"\nReading reference: {args.reference}")
    reference = read_annotation(args.reference)
    print(f"Reading prediction: {args.prediction}")
    prediction = read_annotation(args.prediction)

    result = calculate_matches(reference, prediction)
    print_summary(result)

    if matched_output:
        write_matches(matched_output, result)
        print(f"\nMatched ORF pairs written to: {matched_output}")
    if args.unmatched_output:
        write_unmatched(args.unmatched_output, result)
        print(f"Unmatched gene list written to: {args.unmatched_output}")


if __name__ == "__main__":
    main()
