"""Dependency-light FASTA text I/O shared by prediction and evidence tools."""

from __future__ import annotations

import bz2
import gzip
import sys
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TextIO


@dataclass(frozen=True)
class FastaEntry:
    """A FASTA record retaining its complete header and sequence line layout."""

    header: str
    sequence_lines: tuple[str, ...]

    @property
    def identifier(self) -> str:
        fields = self.header[1:].split() if self.header.startswith(">") else self.header.split()
        return fields[0] if fields else ""

    @property
    def description(self) -> str:
        text = self.header[1:] if self.header.startswith(">") else self.header
        parts = text.split(None, 1)
        return parts[1] if len(parts) == 2 else ""

    @property
    def sequence(self) -> str:
        return "".join(line.strip() for line in self.sequence_lines if line.strip())


@contextmanager
def open_fasta_text(path: str | None, mode: str = "rt") -> Iterator[TextIO]:
    """Open plain, gzip, bzip2, or standard-input FASTA text consistently."""
    if not path or path == "-":
        yield sys.stdin
        return
    kwargs = {} if "b" in mode else {"encoding": "utf-8"}
    if str(path).endswith(".gz"):
        with gzip.open(path, mode, **kwargs) as handle:
            yield handle
        return
    if str(path).endswith(".bz2"):
        with bz2.open(path, mode, **kwargs) as handle:
            yield handle
        return
    with open(path, mode, **kwargs) as handle:
        yield handle


def iter_fasta_entries(lines: Iterable[str]) -> Iterator[FastaEntry]:
    """Yield FASTA records from a line iterable, skipping blank lines."""
    header = ""
    sequence_lines: list[str] = []
    for raw_line in lines:
        line = raw_line.rstrip("\n")
        if not line:
            continue
        if line.startswith(">"):
            if header:
                yield FastaEntry(header, tuple(sequence_lines))
            header = line
            sequence_lines = []
        elif header:
            sequence_lines.append(line)
    if header:
        yield FastaEntry(header, tuple(sequence_lines))
