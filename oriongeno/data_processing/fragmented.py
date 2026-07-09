"""Helpers for fragmented assemblies with many short scaffolds."""

from __future__ import annotations

import copy
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class PackedSegment:
    packed_name: str
    original_name: str
    packed_start: int
    packed_end: int
    original_length: int


@dataclass(frozen=True)
class MappedInterval:
    seq_name: str
    start: int
    end: int
    source_length: int


class FragmentedAssemblyMapper:
    """Map intervals from packed pseudo-contigs back to original scaffolds."""

    def __init__(
        self,
        segments_by_packed: Optional[Dict[str, List[PackedSegment]]] = None,
        original_lengths: Optional[Dict[str, int]] = None,
        packed_lengths: Optional[Dict[str, int]] = None,
        summary: Optional[dict] = None,
    ):
        self.segments_by_packed = segments_by_packed or {}
        self.original_lengths = original_lengths or {}
        self.packed_lengths = packed_lengths or {}
        self.summary = summary or {"mode": "native"}

    @property
    def has_packing(self) -> bool:
        return bool(self.segments_by_packed)

    def map_interval(self, seq_name: str, start: int, end: int) -> Optional[MappedInterval]:
        start = int(start)
        end = int(end)
        if end < start:
            return None
        segments = self.segments_by_packed.get(seq_name)
        if not segments:
            return MappedInterval(
                seq_name=seq_name,
                start=start,
                end=end,
                source_length=self.original_lengths.get(seq_name, end),
            )

        for segment in segments:
            if start < segment.packed_start:
                break
            if start >= segment.packed_start and end <= segment.packed_end:
                offset = start - segment.packed_start
                return MappedInterval(
                    seq_name=segment.original_name,
                    start=offset + 1,
                    end=offset + (end - start + 1),
                    source_length=segment.original_length,
                )
        return None

    def source_names_for_interval(self, seq_name: str, start: int, end: int) -> set[str]:
        """Return original scaffold names touched by an interval.

        Packed pseudo-contigs include artificial N spacers. If a prediction
        crosses a spacer, it cannot be represented as one biological interval,
        but the overlapping source scaffolds can be re-predicted independently.
        """
        start = int(start)
        end = int(end)
        if end < start:
            return set()
        segments = self.segments_by_packed.get(seq_name)
        if not segments:
            return {seq_name}
        touched = set()
        for segment in segments:
            if segment.packed_end < start:
                continue
            if segment.packed_start > end:
                break
            touched.add(segment.original_name)
        return touched

    def remap_annotation(self, annotation) -> dict:
        """Mutate an annotation from packed coordinates back to original scaffolds."""
        if not self.has_packing:
            return {
                "remapped_transcripts": 0,
                "dropped_packed_transcripts": 0,
                "recheck_sequences": [],
            }

        from ..genome_annotation import Transcript

        remapped = 0
        dropped = 0
        recheck_sequences = set()
        output_transcripts = {}

        for transcript_id, transcript in annotation.transcripts.items():
            if transcript.chr not in self.segments_by_packed:
                output_transcripts[transcript_id] = transcript
                continue

            mapped_seq_name = ""
            mapped_lines = []
            invalid = False
            touched_sources = set()
            for lines in transcript.transcript_lines.values():
                for line in lines:
                    touched_sources.update(
                        self.source_names_for_interval(line[0], line[3], line[4])
                    )
                    mapped = self.map_interval(line[0], line[3], line[4])
                    if mapped is None:
                        invalid = True
                        break
                    if mapped_seq_name and mapped.seq_name != mapped_seq_name:
                        invalid = True
                        break
                    mapped_seq_name = mapped.seq_name
                    mapped_line = list(line)
                    mapped_line[0] = mapped.seq_name
                    mapped_line[3] = mapped.start
                    mapped_line[4] = mapped.end
                    mapped_lines.append(mapped_line)
                if invalid:
                    break

            if invalid or not mapped_seq_name or not mapped_lines:
                recheck_sequences.update(touched_sources)
                dropped += 1
                continue

            new_transcript = Transcript(
                transcript.id,
                transcript.gene_id,
                mapped_seq_name,
                transcript.source_anno,
                transcript.strand,
            )
            for mapped_line in mapped_lines:
                new_transcript.add_line(mapped_line)
            output_transcripts[transcript_id] = new_transcript
            remapped += 1

        annotation.transcripts = output_transcripts
        annotation.genes = {"None": []}
        annotation.gene_gtf = {}
        return {
            "remapped_transcripts": remapped,
            "dropped_packed_transcripts": dropped,
            "recheck_sequences": sorted(recheck_sequences),
        }


def _copy_record(seqrec):
    copied = copy.copy(seqrec)
    copied.annotations = dict(getattr(seqrec, "annotations", {}) or {})
    copied.features = list(getattr(seqrec, "features", []) or [])
    copied.dbxrefs = list(getattr(seqrec, "dbxrefs", []) or [])
    return copied


def _make_seq_record(name: str, sequence: str):
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    return SeqRecord(Seq(sequence), id=name, name=name, description=name)


def _unique_name(base_name: str, used_names: set[str]) -> str:
    if base_name not in used_names:
        used_names.add(base_name)
        return base_name
    suffix = 2
    while True:
        candidate = f"{base_name}_{suffix}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        suffix += 1


def _iter_records(genome) -> Iterable[Tuple[str, object, int]]:
    for name, seqrec in genome.items():
        yield str(name), seqrec, len(seqrec.seq)


def prepare_inference_genome(
    genome,
    *,
    assembly_mode: str,
    seq_len: int,
    flank_size: int,
    min_seq_len: int,
    fragmented_record_threshold: int,
    pack_threshold: int,
    pack_spacer_len: int,
    pack_target_size: int,
):
    """Return a genome optimized for inference and a coordinate mapper.

    In ``auto`` mode, packing is enabled only when the number of retained FASTA
    records exceeds ``fragmented_record_threshold``. Long sequences are kept in
    their native coordinates; short sequences are concatenated into pseudo-contigs
    separated by N spacers and later remapped.
    """
    if assembly_mode not in {"auto", "native", "packed"}:
        raise ValueError(f"Unsupported assembly mode: {assembly_mode}")

    original_lengths = {name: length for name, _, length in _iter_records(genome)}
    records = [
        (name, seqrec, length)
        for name, seqrec, length in _iter_records(genome)
        if length >= int(min_seq_len)
    ]
    skipped_by_length = len(original_lengths) - len(records)

    threshold = int(pack_threshold or 0)
    if threshold <= 0:
        threshold = int(seq_len)
    spacer_len = int(pack_spacer_len or 0)
    if spacer_len < 0:
        raise ValueError("pack_spacer_len must be non-negative.")
    target_size = int(pack_target_size or 0)
    if target_size <= 0:
        target_size = max(int(seq_len), 1)

    short_records = [record for record in records if record[2] < threshold]
    should_pack = assembly_mode == "packed" or (
        assembly_mode == "auto"
        and len(records) > int(fragmented_record_threshold)
        and bool(short_records)
    )

    summary = {
        "mode": "packed" if should_pack else "native",
        "assembly_mode": assembly_mode,
        "input_sequences": len(original_lengths),
        "kept_sequences": len(records),
        "skipped_by_min_seq_len": skipped_by_length,
        "native_sequences": len(records),
        "packed_source_sequences": 0,
        "packed_records": 0,
        "pack_threshold": threshold,
        "pack_spacer_len": spacer_len,
        "pack_target_size": target_size,
        "fragmented_record_threshold": int(fragmented_record_threshold),
    }

    if not should_pack:
        return OrderedDict((name, seqrec) for name, seqrec, _ in records), FragmentedAssemblyMapper(
            original_lengths=original_lengths,
            summary=summary,
        )

    output_genome = OrderedDict()
    segments_by_packed: Dict[str, List[PackedSegment]] = {}
    packed_lengths: Dict[str, int] = {}
    used_names = set(original_lengths)
    pack_index = 1
    current_name = ""
    current_parts: List[str] = []
    current_segments: List[PackedSegment] = []
    current_len = 0
    packed_source_sequences = 0
    native_sequences = 0

    def flush_pack():
        nonlocal current_name, current_parts, current_segments, current_len
        if not current_parts:
            return
        output_genome[current_name] = _make_seq_record(current_name, "".join(current_parts))
        segments_by_packed[current_name] = current_segments
        packed_lengths[current_name] = current_len
        current_name = ""
        current_parts = []
        current_segments = []
        current_len = 0

    def start_pack():
        nonlocal current_name, pack_index
        current_name = _unique_name(f"packed_scaffold_{pack_index:06d}", used_names)
        pack_index += 1

    for name, seqrec, length in records:
        if length >= threshold:
            flush_pack()
            output_genome[name] = _copy_record(seqrec)
            native_sequences += 1
            continue

        sequence = str(seqrec.seq)
        projected_len = current_len + (spacer_len if current_parts else 0) + length
        if current_parts and projected_len > target_size:
            flush_pack()
        if not current_parts:
            start_pack()
        elif spacer_len > 0:
            current_parts.append("N" * spacer_len)
            current_len += spacer_len

        packed_start = current_len + 1
        current_parts.append(sequence)
        current_len += length
        current_segments.append(
            PackedSegment(
                packed_name=current_name,
                original_name=name,
                packed_start=packed_start,
                packed_end=current_len,
                original_length=length,
            )
        )
        packed_source_sequences += 1

    flush_pack()

    summary.update(
        {
            "native_sequences": native_sequences,
            "packed_source_sequences": packed_source_sequences,
            "packed_records": len(segments_by_packed),
            "inference_sequences": len(output_genome),
            "input_bases": sum(original_lengths.values()),
            "inference_bases": sum(len(seqrec.seq) for seqrec in output_genome.values()),
        }
    )
    logging.info(
        "Fragmented assembly packing: %s source scaffolds -> %s packed records; "
        "%s native records kept.",
        packed_source_sequences,
        len(segments_by_packed),
        native_sequences,
    )
    return output_genome, FragmentedAssemblyMapper(
        segments_by_packed=segments_by_packed,
        original_lengths=original_lengths,
        packed_lengths=packed_lengths,
        summary=summary,
    )
