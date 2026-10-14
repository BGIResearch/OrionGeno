"""Helpers for comparing stop-inclusive and stop-exclusive CDS annotations."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable


CdsSegment = tuple[str, str, int, int, str]
Interval = tuple[str, str, int, int]


def merge_stop_codons_into_cds(
    cds_segments: Iterable[CdsSegment],
    stop_codons: Iterable[Interval],
) -> list[CdsSegment]:
    """Return a stop-inclusive, interval-normalized CDS structure.

    RefSeq GTF commonly stores the terminal stop codon in separate
    ``stop_codon`` rows, while OrionGeno includes those bases in ``CDS``.  This
    function unions both feature types without changing already stop-inclusive
    annotations.  CDS phase is retained from the first CDS block in transcript
    direction; a stop-only split block is left unphased.
    """
    cds = list(cds_segments)
    if not cds:
        return []
    by_molecule: dict[tuple[str, str], list[tuple[int, int, str, bool]]] = defaultdict(list)
    for chrom, strand, start, end, phase in cds:
        by_molecule[(chrom, strand)].append((start, end, phase, True))
    for chrom, strand, start, end in stop_codons:
        by_molecule[(chrom, strand)].append((start, end, ".", False))

    normalized: list[CdsSegment] = []
    for (chrom, strand), pieces in sorted(by_molecule.items()):
        merged: list[list[tuple[int, int, str, bool]]] = []
        for piece in sorted(pieces, key=lambda item: (item[0], item[1], not item[3])):
            if merged and piece[0] <= max(item[1] for item in merged[-1]) + 1:
                merged[-1].append(piece)
            else:
                merged.append([piece])
        for group in merged:
            cds_pieces = [piece for piece in group if piece[3]]
            if not cds_pieces:
                phase = "."
            elif strand == "-":
                phase = max(cds_pieces, key=lambda item: (item[1], item[0]))[2]
            else:
                phase = min(cds_pieces, key=lambda item: (item[0], item[1]))[2]
            normalized.append(
                (
                    chrom,
                    strand,
                    min(item[0] for item in group),
                    max(item[1] for item in group),
                    phase,
                )
            )
    return sorted(normalized, key=lambda item: (item[0], item[1], item[2], item[3], item[4]))
