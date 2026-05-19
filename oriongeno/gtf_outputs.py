"""GTF filtering and repeat-output helpers."""

from __future__ import annotations

import copy
import logging
import os

import numpy as np

def repeat_output_path(output_base):
    """Return the user-facing repeat GTF path beside the requested output."""
    root, ext = os.path.splitext(os.fspath(output_base))
    if ext:
        return f"{root}.repeat{ext}"
    return f"{os.fspath(output_base)}.repeat.gtf"


def _write_prediction_gtf(transcripts, output_path, id_prefix):
    from .genome_anno import Anno

    annotation = Anno("", "oriongeno")
    annotation.add_transcripts(copy.deepcopy(transcripts))
    annotation.norm_tx_format()
    annotation.find_genes()
    annotation.rename_tx_ids(id_prefix)
    annotation.write_anno(output_path)
    return annotation


class RepeatGtfWriter:
    """Collect raw repeat-head labels and write positive runs as GTF rows."""

    def __init__(
        self,
        output_base,
        id_prefix="",
        coordinate_mapper=None,
    ):
        self.output_path = repeat_output_path(output_base)
        self.id_prefix = id_prefix
        self.coordinate_mapper = coordinate_mapper
        self.records = []
        self._last_by_key = {}
        self.recheck_sequences = set()

    def _append_interval(self, seq_name, strand, start, end):
        if end < start:
            return
        key = (seq_name, strand)
        last = self._last_by_key.get(key)
        if last is not None and start <= last["end"] + 1:
            last["end"] = max(last["end"], end)
            return
        record = {
            "seq_name": seq_name,
            "strand": strand,
            "start": start,
            "end": end,
        }
        self.records.append(record)
        self._last_by_key[key] = record

    def add_predictions(self, repeat_labels, coords, strand, genome_seq_dict):
        labels = np.asarray(repeat_labels)
        chunk_coords = list(coords)
        if strand == "-":
            labels = labels[::-1, ::-1]
            chunk_coords = chunk_coords[::-1]

        for chunk_labels, coord in zip(labels, chunk_coords):
            seq_name = coord[0]
            start = int(coord[2])
            end = int(coord[3])
            seq_len = genome_seq_dict.get(seq_name)
            if seq_len is None:
                valid_start = start
                valid_end = end
            else:
                valid_start = max(1, start)
                valid_end = min(end, int(seq_len))
            if valid_end < valid_start:
                continue

            label_start = valid_start - start
            label_end = label_start + (valid_end - valid_start + 1)
            valid_labels = np.asarray(chunk_labels[label_start:label_end], dtype=np.int8)
            if valid_labels.size == 0:
                continue

            positive = valid_labels == 1
            if not positive.any():
                continue
            change_points = np.where(np.diff(positive.astype(np.int8)) != 0)[0]
            run_starts = np.insert(change_points + 1, 0, 0)
            run_ends = np.append(change_points, positive.size - 1)
            for run_start, run_end in zip(run_starts, run_ends):
                if not positive[run_start]:
                    continue
                interval_start = valid_start + int(run_start)
                interval_end = valid_start + int(run_end)
                if self.coordinate_mapper is not None:
                    mapped = self.coordinate_mapper.map_interval(
                        seq_name,
                        interval_start,
                        interval_end,
                    )
                    if mapped is None:
                        source_names = self.coordinate_mapper.source_names_for_interval(
                            seq_name,
                            interval_start,
                            interval_end,
                        )
                        self.recheck_sequences.update(source_names)
                        continue
                    self._append_interval(
                        mapped.seq_name,
                        strand,
                        mapped.start,
                        mapped.end,
                    )
                    continue
                self._append_interval(
                    seq_name,
                    strand,
                    interval_start,
                    interval_end,
                )

    def remove_sequences(self, seq_names):
        remove = set(seq_names)
        if not remove:
            return
        self.records = [
            record for record in self.records if record["seq_name"] not in remove
        ]
        self._last_by_key = {}
        for record in self.records:
            self._last_by_key[(record["seq_name"], record["strand"])] = record

    def consume_recheck_sequences(self):
        seq_names = sorted(self.recheck_sequences)
        self.recheck_sequences.clear()
        return seq_names

    def write(self):
        output_dir = os.path.dirname(os.path.abspath(self.output_path))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        sorted_records = sorted(
            self.records,
            key=lambda record: (
                record["seq_name"],
                record["strand"],
                record["start"],
                record["end"],
            ),
        )
        with open(self.output_path, "w", encoding="utf-8") as file_obj:
            for index, record in enumerate(sorted_records, start=1):
                repeat_id = f"{self.id_prefix}r{index}" if self.id_prefix else f"r{index}"
                attributes = f'repeat_id "{repeat_id}";'
                fields = [
                    record["seq_name"],
                    "OrionGeno",
                    "repeat_region",
                    str(record["start"]),
                    str(record["end"]),
                    ".",
                    record["strand"],
                    ".",
                    attributes,
                ]
                file_obj.write("\t".join(fields) + "\n")
        logging.info("Repeat GTF records: %s", len(sorted_records))
        logging.info("Repeat GTF output: %s", self.output_path)
        return self.output_path


def _assemble_transcript(exons, sequence, strand):
    from Bio.Seq import Seq

    parts = []
    exons.sort(reverse=strand == "-")
    for exon in exons:
        exon_seq = sequence.seq[exon[0] - 1 : exon[1]]
        if strand == "-":
            exon_seq = exon_seq.reverse_complement()
        parts.append(str(exon_seq))

    coding_seq = Seq("".join(parts))
    if len(coding_seq) > 0 and len(coding_seq) % 3 == 0:
        protein_seq = coding_seq.translate()
        if protein_seq[-1] == "*":
            return coding_seq, protein_seq
    return None, None


def _has_internal_stop(protein_seq):
    if protein_seq is None:
        return False
    return "*" in protein_seq[:-1]


def filter_and_write_outputs(
    annotation,
    genome,
    genome_seq_dict,
    output_path,
    id_prefix,
):
    from tqdm import tqdm

    output_transcripts = {}

    for transcript_id, transcript in tqdm(
        annotation.transcripts.items(),
        desc="OrionGeno gene filtering",
    ):
        exons = transcript.get_type_coords("CDS", frame=False)
        coding_seq, protein_seq = _assemble_transcript(
            exons,
            genome[transcript.chr],
            transcript.strand,
        )
        filtered_by_stop = not coding_seq or _has_internal_stop(protein_seq)
        filtered_by_length = transcript.get_cds_len() < 61
        filtered_by_bounds = (
            transcript.start < 1
            or transcript.end > genome_seq_dict[transcript.chr]
        )

        if filtered_by_bounds or transcript.get_cds_len() <= 0:
            continue

        if filtered_by_length or filtered_by_stop:
            continue
        output_transcripts[transcript_id] = transcript

    return _write_prediction_gtf(output_transcripts, output_path, id_prefix)
