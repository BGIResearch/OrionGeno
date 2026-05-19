"""FASTA loading and record-count helpers."""

from __future__ import annotations

import bz2
import gzip
import logging


def load_genome(genome_path):
    from Bio import SeqIO

    if genome_path.endswith(".gz"):
        with gzip.open(genome_path, "rt") as file:
            return SeqIO.to_dict(SeqIO.parse(file, "fasta"))
    if genome_path.endswith(".bz2"):
        with bz2.open(genome_path, "rt") as file:
            return SeqIO.to_dict(SeqIO.parse(file, "fasta"))
    with open(genome_path, "r") as file:
        return SeqIO.to_dict(SeqIO.parse(file, "fasta"))


def count_fasta_records(genome_path):
    opener = open
    if genome_path.endswith(".gz"):
        opener = gzip.open
    elif genome_path.endswith(".bz2"):
        opener = bz2.open

    with opener(genome_path, "rt") as file_obj:
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
