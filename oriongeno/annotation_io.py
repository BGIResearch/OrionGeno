"""Shared parsing and serialization helpers for GFF3 and GTF attributes.

Annotation helpers operate on both GFF3's ``key=value`` syntax and GTF's
``key "value"`` syntax.  Keeping the token representation here prevents
individual workflow stages from silently interpreting the same ninth column
differently.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
import gzip
import os
from urllib.parse import quote, unquote


GTF_GENE_ID = "gene_id"
GTF_TRANSCRIPT_ID = "transcript_id"
GFF3_ATTRIBUTE_SAFE = ":._+-"
GFF3_LIST_ATTRIBUTES = {"Parent", "Alias", "Note", "Dbxref", "Ontology_term"}


def open_annotation_text(path, mode="r"):
    """Open UTF-8 annotation text, compressing/decompressing a .gz suffix."""
    opener = gzip.open if os.fspath(path).lower().endswith(".gz") else open
    return opener(path, mode.replace("t", "") + "t", encoding="utf-8", newline="")


@dataclass
class AttributeToken:
    """One parsed ninth-column attribute, preserving its input style."""

    key: str
    value: str = ""
    separator: str = ""
    quoted: bool = False


def split_attributes(attributes: str) -> list[AttributeToken]:
    """Parse GFF3/GTF attributes without discarding order or duplicate keys."""
    text = str(attributes).strip()
    if not text or text == ".":
        return []

    tokens: list[AttributeToken] = []
    for raw_item in text.rstrip(";").split(";"):
        item = raw_item.strip()
        if not item:
            continue

        equal_index = item.find("=")
        first_space = min(
            [index for index in (item.find(" "), item.find("\t")) if index >= 0],
            default=-1,
        )
        if equal_index >= 0 and (first_space < 0 or equal_index < first_space):
            key, value = item.split("=", 1)
            separator = "="
        else:
            parts = item.split(None, 1)
            if len(parts) == 1:
                tokens.append(AttributeToken(parts[0]))
                continue
            key, value = parts
            separator = " "

        value = value.strip()
        quoted = len(value) >= 2 and value[0] == '"' and value[-1] == '"'
        if quoted:
            value = value[1:-1]
        tokens.append(AttributeToken(key=key.strip(), value=value, separator=separator, quoted=quoted))
    return tokens


def attribute_values(attributes: str | Iterable[AttributeToken]) -> dict[str, list[str]]:
    """Return every value for each key, preserving duplicate attributes."""
    tokens = split_attributes(attributes) if isinstance(attributes, str) else attributes
    values: dict[str, list[str]] = {}
    for token in tokens:
        if token.separator:
            values.setdefault(token.key, []).append(token.value)
    return values


def parse_attributes(attributes: str, *, keep: str = "last") -> dict[str, str]:
    """Return a convenient single-value view of GFF3/GTF attributes.

    ``keep`` controls duplicate-key handling.  Most pre-existing helpers used
    last-value-wins semantics, while callers that need all values should use
    :func:`attribute_values` instead.
    """
    if keep not in {"first", "last"}:
        raise ValueError("keep must be 'first' or 'last'")
    parsed: dict[str, str] = {}
    for token in split_attributes(attributes):
        if not token.separator or (keep == "first" and token.key in parsed):
            continue
        parsed[token.key] = token.value
    return parsed


def format_attributes(tokens: Iterable[AttributeToken]) -> str:
    """Serialize tokens while retaining each token's GFF3 or GTF style."""
    rendered: list[str] = []
    for token in tokens:
        if not token.separator:
            rendered.append(token.key)
        elif token.separator == "=":
            rendered.append(f"{token.key}={token.value}")
        else:
            value = f'"{token.value}"' if token.quoted else token.value
            rendered.append(f"{token.key} {value}")
    return "; ".join(rendered) + ";" if rendered else "."


def format_gtf_attributes(
    gene_id: str = "",
    transcript_id: str = "",
    extra_attributes: Iterable[str] | None = None,
) -> str:
    """Serialize the standard GTF identifiers plus already-rendered extras."""
    parts: list[str] = []
    if gene_id:
        parts.append(f'{GTF_GENE_ID} "{gene_id}";')
    if transcript_id:
        parts.append(f'{GTF_TRANSCRIPT_ID} "{transcript_id}";')
    if extra_attributes:
        parts.extend(str(item).strip() for item in extra_attributes if str(item).strip())
    return " ".join(parts)


def format_gff3_attributes(items: Iterable[tuple[str, object]]) -> str:
    """Serialize GFF3 attributes with standards-safe value escaping."""
    rendered: list[str] = []
    for key, value in items:
        if value in (None, ""):
            continue
        if isinstance(value, (list, tuple)):
            text = ",".join(quote(str(item), safe=GFF3_ATTRIBUTE_SAFE) for item in value if str(item))
        else:
            text = quote(str(value), safe=GFF3_ATTRIBUTE_SAFE)
        if text:
            rendered.append(f"{key}={text}")
    return ";".join(rendered) if rendered else "."


def format_parsed_gff3_attributes(items: Iterable[tuple[str, str]]) -> str:
    """Re-serialize values from parse_attributes without double escaping.

    The compatibility parser retains wire-format escapes. Split reserved lists
    before decoding so an escaped comma inside one ID remains part of that ID.
    New, unescaped values should use format_gff3_attributes instead.
    """
    return format_gff3_attributes(
        (
            key,
            [unquote(part) for part in value.split(",")]
            if key in GFF3_LIST_ATTRIBUTES
            else unquote(value),
        )
        for key, value in items
    )
