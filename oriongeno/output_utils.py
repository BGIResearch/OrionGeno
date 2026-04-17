"""Output path and post-processing helpers for OrienGeno inference.

Authors: wangshengfu, caixudong
"""

import copy
import logging
import os
import sys

from Bio.Seq import Seq

from .genome_anno import Anno, UTR_FEATURE_TYPES, format_gtf_attributes


def assemble_transcript(exons, sequence, strand):
    """Build a coding sequence and translated protein from exon coordinates."""
    parts = []
    exons.sort(reverse=strand == "-")
    for exon in exons:
        exon_seq = sequence.seq[exon[0] - 1 : exon[1]]
        if strand == "-":
            exon_seq = exon_seq.reverse_complement()
        parts.append(str(exon_seq))

    coding_seq = Seq("".join(parts))
    if len(coding_seq) > 0 and len(coding_seq) % 3 == 0:
        prot_seq = coding_seq.translate()
        if prot_seq[-1] == "*":
            return coding_seq, prot_seq
    return None, None


def check_in_frame_stop_codons(seq):
    """Return True when a translated protein contains an internal stop codon."""
    if seq is None:
        return False
    return "*" in seq[:-1]


def ensure_parent_dir(path):
    """Create the parent directory for a file path when it is missing."""
    parent_dir = os.path.dirname(os.path.abspath(path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def build_variant_output_path(out_path, suffix):
    """Append a suffix before the output extension while keeping absolute paths."""
    base, ext = os.path.splitext(out_path)
    if ext:
        return f"{base}.{suffix}{ext}"
    return f"{out_path}.{suffix}"


def resolve_output_paths(out_path, repeat_out_path, output_gene, output_repeat):
    """Resolve final gene/repeat output paths from a shared base path."""
    gene_out = ""
    resolved_repeat_out = ""

    if output_gene:
        gene_out = build_variant_output_path(out_path, "gene")
    if output_repeat:
        resolved_repeat_out = (
            os.path.abspath(repeat_out_path)
            if repeat_out_path
            else build_variant_output_path(out_path, "repeat")
        )

    return gene_out, resolved_repeat_out


def resolve_model_bundle_paths(model_path):
    """Resolve gene/repeat checkpoint paths from one user-facing model path."""
    model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        logging.error(f"Error: The file '{model_path}' does not exist.")
        sys.exit(1)
    if not os.path.isdir(model_path):
        logging.error(
            "ERROR: The model path must be a valid OrionGeno model directory. "
            f"Current value: {model_path}"
        )
        sys.exit(1)

    gene_model_path = os.path.join(model_path, "gene.bin")
    repeat_model_path = os.path.join(model_path, "repeat.bin")
    return {
        "bundle_dir": model_path,
        "gene_model_path": gene_model_path if os.path.exists(gene_model_path) else "",
        "repeat_model_path": (
            repeat_model_path if os.path.exists(repeat_model_path) else ""
        ),
        "gene_checkpoint_format": "runtime_gene",
        "repeat_checkpoint_format": "runtime_repeat",
        "gene_config_path": "",
        "repeat_config_path": "",
        "repeat_classifier_path": "",
    }


def merge_repeat_intervals(intervals):
    """Merge adjacent repeat intervals with the same label."""
    if not intervals:
        return []

    merged = []
    for seq_name, start, end, label in sorted(
        intervals,
        key=lambda row: (row[0], row[1], row[2], row[3]),
    ):
        if (
            merged
            and merged[-1][0] == seq_name
            and merged[-1][3] == label
            and start <= merged[-1][2] + 1
        ):
            merged[-1][2] = max(merged[-1][2], end)
        else:
            merged.append([seq_name, start, end, label])
    return [tuple(row) for row in merged]


def fill_short_nonrepeat_gaps(intervals, max_gap_bases=30):
    """Fill short non-repeat runs that are bracketed by repeat-positive runs."""
    if int(max_gap_bases) < 1 or not intervals:
        return intervals

    smoothed = [list(row) for row in intervals]
    for index in range(1, len(smoothed) - 1):
        prev_interval = smoothed[index - 1]
        current_interval = smoothed[index]
        next_interval = smoothed[index + 1]

        if (
            prev_interval[0] != current_interval[0]
            or current_interval[0] != next_interval[0]
        ):
            continue
        if int(prev_interval[3]) != 1 or int(current_interval[3]) != 0:
            continue
        if int(next_interval[3]) != 1:
            continue

        gap_length = int(current_interval[2]) - int(current_interval[1]) + 1
        if gap_length <= int(max_gap_bases):
            current_interval[3] = 1

    return [tuple(row) for row in smoothed]


def filter_short_repeat_intervals(intervals, min_repeat_bases=1):
    """Drop repeat-positive runs shorter than the requested threshold."""
    if int(min_repeat_bases) <= 1:
        return intervals

    filtered = []
    for seq_name, start, end, label in intervals:
        interval_len = int(end) - int(start) + 1
        if int(label) == 1 and interval_len < int(min_repeat_bases):
            continue
        filtered.append((seq_name, start, end, label))
    return filtered


def build_repeat_gtf_attributes(repeat_id):
    """Build one RepeatMasker-like GTF attribute string for a repeat interval."""
    return format_gtf_attributes(
        [
            ("ID", repeat_id),
            ("repeat_id", repeat_id),
            ("Name", "Motif:unknown"),
            ("repeat_name", "unknown"),
            ("repeat_class", "unknown"),
            ("repeat_family", "unknown"),
            ("repeat_label", "1"),
        ]
    )


def build_gene_gtf_header_lines(include_utr=True):
    """Build header comment lines for the gene annotation GTF."""
    return [
        "# OrionGeno gene annotation",
        "# format: GTF",
        f"# utr_included: {'True' if include_utr else 'False'}",
    ]


def build_repeat_gtf_header_lines():
    """Build header comment lines for the repeat annotation GTF."""
    return [
        "# OrionGeno repeat annotation",
        "# format: GTF",
        "# style: RepeatMasker-like layout written in GTF syntax",
        "# feature_type: dispersed_repeat",
        "# repeat_label: 1=repeat, 0=non-repeat",
        "# repeat_name/repeat_class/repeat_family are written as unknown because OrionGeno currently predicts repeat intervals only.",
        "# strand: +",
        "# strand_note: '+' is a fixed OrionGeno output convention for strand-independent repeat intervals and is not a RepeatMasker consensus-orientation field.",
    ]


def write_repeat_outputs(
    intervals,
    repeat_out,
    min_repeat_bases=1,
    max_nonrepeat_gap_bases=30,
):
    """Write merged repeat annotation intervals as GTF rows for repeat-positive runs."""
    ensure_parent_dir(repeat_out)
    merged_intervals = merge_repeat_intervals(intervals)
    merged_intervals = fill_short_nonrepeat_gaps(
        merged_intervals,
        max_gap_bases=max_nonrepeat_gap_bases,
    )
    merged_intervals = merge_repeat_intervals(merged_intervals)
    merged_intervals = filter_short_repeat_intervals(
        merged_intervals,
        min_repeat_bases=min_repeat_bases,
    )

    with open(repeat_out, "w", encoding="utf-8") as handle:
        for header_line in build_repeat_gtf_header_lines():
            handle.write(f"{header_line}\n")
        repeat_index = 0
        for seq_name, start, end, label in merged_intervals:
            if int(label) != 1:
                continue
            repeat_index += 1
            repeat_id = f"repeat_{repeat_index:08d}"
            attributes = build_repeat_gtf_attributes(repeat_id)
            fields = [
                seq_name,
                "OrienGeno",
                "dispersed_repeat",
                str(start),
                str(end),
                ".",
                "+",
                ".",
                attributes,
            ]
            handle.write("\t".join(fields) + "\n")


def namespace_transcripts(transcripts, namespace):
    """Make transcript and gene identifiers unique before cross-rank merging."""
    renamed = {}
    for transcript in transcripts.values():
        transcript.id = f"{namespace}_{transcript.id}"
        transcript.gene_id = f"{namespace}_{transcript.gene_id}"
        renamed[transcript.id] = transcript
    return renamed


def write_filtered_gene_outputs(
    anno,
    genome,
    genome_seq_dict,
    gtf_out,
    id_prefix,
    include_utr=True,
    quiet_logs=False,
):
    """Apply the strict post-filtering rules and write one final GTF."""
    if not quiet_logs:
        logging.info(
            f"Total gene transcripts before final filtering: {len(anno.transcripts)}"
        )

    anno_outp_strict = Anno("", "anno")
    out_tx_strict = {}
    for tx_id, tx in anno.transcripts.items():
        exons = tx.get_type_coords("CDS", frame=False)
        filt_stop = False
        filt_length = False
        filt_index = False

        coding_seq, prot_seq = assemble_transcript(exons, genome[tx.chr], tx.strand)
        if not coding_seq or check_in_frame_stop_codons(prot_seq):
            filt_stop = True
        if tx.get_cds_len() < 61:
            filt_length = True

        chrom_len = genome_seq_dict.get(tx.chr)
        if chrom_len is None:
            filt_index = True
        elif tx.start < 1 or tx.end > chrom_len:
            filt_index = True

        if not filt_index and tx.get_cds_len() > 0 and not filt_length and not filt_stop:
            out_tx_strict[tx_id] = tx

    anno_outp_strict.add_transcripts(copy.deepcopy(out_tx_strict))
    if not include_utr:
        anno_outp_strict.remove_feature_types(UTR_FEATURE_TYPES)
    anno_outp_strict.norm_tx_format()
    anno_outp_strict.find_genes()
    anno_outp_strict.rename_tx_ids(id_prefix)
    anno_outp_strict.write_anno(
        gtf_out,
        include_utr=include_utr,
        header_lines=build_gene_gtf_header_lines(include_utr=include_utr),
    )
