"""FASTA loading and record-count helpers."""

from __future__ import annotations

import logging

from ..fasta_io import open_fasta_text


def load_genome(genome_path):
    from Bio import SeqIO

    with open_fasta_text(genome_path) as file:
        return SeqIO.to_dict(SeqIO.parse(file, "fasta"))


def count_fasta_records(genome_path):
    with open_fasta_text(genome_path) as file_obj:
        return sum(1 for line in file_obj if line.startswith(">"))


def should_skip_by_fasta_record_count(genome_path, max_fasta_records):
    if max_fasta_records is None or max_fasta_records <= 0:
        return False
    record_count = count_fasta_records(genome_path)
    logging.info("FASTA records: %s", record_count)
    if record_count <= max_fasta_records:
        return False
    logging.warning(
        "Skip inference because FASTA records=%s exceeds max-fasta-records=%s.",
        record_count,
        max_fasta_records,
    )
    return True
