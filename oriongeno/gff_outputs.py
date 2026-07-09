"""Gene filtering and GFF-oriented annotation output helpers."""

from __future__ import annotations

import copy
import logging
import os

import numpy as np

from .constants import DEFAULT_GENE_FILTER_MODE


def annotation_output_is_gff(output_path):
    path_text = os.fspath(output_path).lower()
    if path_text.endswith(".gz"):
        path_text = path_text[:-3]
    return os.path.splitext(path_text)[1] in {".gff", ".gff3"}


def annotation_output_format(output_path):
    return "GFF" if annotation_output_is_gff(output_path) else "GTF"


def repeat_output_path(output_base):
    """Return the user-facing repeat annotation path beside the requested output."""
    root, ext = os.path.splitext(os.fspath(output_base))
    if ext:
        return f"{root}.repeat{ext}"
    return f"{os.fspath(output_base)}.repeat.gff"


def _reset_transcript_lines(transcript):
    transcript.transcript_lines = {}
    transcript.start = -1
    transcript.end = -1
    transcript.cds_len = -1
    transcript.cds_coords = {}
    transcript.source_method = ""


def _clip_annotation_line_to_bounds(line, seq_len):
    clipped = list(line)
    start = int(clipped[3])
    end = int(clipped[4])
    if end < 1 or start > seq_len:
        return None
    clipped[3] = max(1, start)
    clipped[4] = min(seq_len, end)
    if clipped[3] > clipped[4]:
        return None
    return clipped


def _clip_transcripts_to_bounds(transcripts, genome_seq_dict):
    regenerated_features = {"gene", "transcript", "intron", "start_codon", "stop_codon"}
    removed_transcripts = []
    removed_features = 0
    clipped_features = 0

    for transcript_id, transcript in list(transcripts.items()):
        seq_len = genome_seq_dict.get(transcript.chr)
        if seq_len is None:
            continue

        original_lines = copy.deepcopy(transcript.transcript_lines)
        _reset_transcript_lines(transcript)
        for feature_type, lines in original_lines.items():
            if feature_type in regenerated_features:
                continue
            for line in lines:
                clipped = _clip_annotation_line_to_bounds(line, int(seq_len))
                if clipped is None:
                    removed_features += 1
                    continue
                if clipped[3] != int(line[3]) or clipped[4] != int(line[4]):
                    clipped_features += 1
                transcript.add_line(clipped)

        if "CDS" not in transcript.transcript_lines and "exon" not in transcript.transcript_lines:
            removed_transcripts.append(transcript_id)
            continue
        transcript.check_splits()
        transcript.redo_phase()

    for transcript_id in removed_transcripts:
        transcripts.pop(transcript_id, None)

    return {
        "clipped_features": clipped_features,
        "removed_features": removed_features,
        "removed_transcripts": len(removed_transcripts),
    }


def _write_prediction_gff(transcripts, output_path, id_prefix, genome_seq_dict=None):
    from .genome_annotation import Anno

    annotation = Anno("", "oriongeno")
    annotation.add_transcripts(copy.deepcopy(transcripts))
    if genome_seq_dict is not None:
        clip_stats = _clip_transcripts_to_bounds(annotation.transcripts, genome_seq_dict)
        if any(clip_stats.values()):
            logging.info(
                "Adjusted out-of-bound annotation coordinates: %s clipped features, "
                "%s removed features, %s removed transcripts.",
                clip_stats["clipped_features"],
                clip_stats["removed_features"],
                clip_stats["removed_transcripts"],
            )
    annotation.norm_tx_format()
    annotation.find_genes()
    annotation.rename_tx_ids(id_prefix)
    if annotation_output_is_gff(output_path):
        annotation.write_gff3(output_path)
    else:
        annotation.write_anno(output_path)
    return annotation


class RepeatGffWriter:
    """Collect raw repeat-head labels and write positive runs as annotation rows."""

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
            if annotation_output_is_gff(self.output_path):
                file_obj.write("##gff-version 3\n")
            for index, record in enumerate(sorted_records, start=1):
                repeat_id = f"{self.id_prefix}r{index}" if self.id_prefix else f"r{index}"
                if annotation_output_is_gff(self.output_path):
                    attributes = f"ID={repeat_id}"
                else:
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
        logging.info("Repeat %s records: %s", annotation_output_format(self.output_path), len(sorted_records))
        logging.info("Repeat %s output: %s", annotation_output_format(self.output_path), self.output_path)
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


def _strict_filter_transcripts(annotation, genome, genome_seq_dict):
    from tqdm import tqdm

    output_transcripts = {}
    total_clip_stats = {
        "clipped_features": 0,
        "removed_features": 0,
        "removed_transcripts": 0,
    }

    for transcript_id, original_transcript in tqdm(
        annotation.transcripts.items(),
        desc="OrionGeno gene filtering",
    ):
        if (
            original_transcript.chr not in genome
            or original_transcript.chr not in genome_seq_dict
        ):
            continue

        transcript_map = {transcript_id: copy.deepcopy(original_transcript)}
        clip_stats = _clip_transcripts_to_bounds(transcript_map, genome_seq_dict)
        for key, value in clip_stats.items():
            total_clip_stats[key] += value
        if transcript_id not in transcript_map:
            continue

        transcript = transcript_map[transcript_id]
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

    if any(total_clip_stats.values()):
        logging.info(
            "Adjusted out-of-bound annotation coordinates before strict filtering: "
            "%s clipped features, %s removed features, %s removed transcripts.",
            total_clip_stats["clipped_features"],
            total_clip_stats["removed_features"],
            total_clip_stats["removed_transcripts"],
        )

    return output_transcripts


def filter_and_write_outputs(
    annotation,
    genome,
    genome_seq_dict,
    output_path,
    id_prefix,
    gene_filter_mode=DEFAULT_GENE_FILTER_MODE,
):
    gene_filter_mode = (gene_filter_mode or DEFAULT_GENE_FILTER_MODE).lower()
    if gene_filter_mode == "none":
        logging.info(
            "Gene %s filtering disabled; writing %s predicted transcripts to %s.",
            annotation_output_format(output_path),
            len(annotation.transcripts),
            output_path,
        )
        return _write_prediction_gff(
            annotation.transcripts,
            output_path,
            id_prefix,
            genome_seq_dict=genome_seq_dict,
        )
    if gene_filter_mode != "strict":
        raise ValueError(f"Unsupported gene_filter_mode: {gene_filter_mode!r}")

    output_transcripts = _strict_filter_transcripts(annotation, genome, genome_seq_dict)
    return _write_prediction_gff(
        output_transcripts,
        output_path,
        id_prefix,
        genome_seq_dict=genome_seq_dict,
    )
