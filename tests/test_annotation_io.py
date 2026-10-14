from __future__ import annotations

from oriongeno.annotation_io import (
    AttributeToken,
    attribute_values,
    format_attributes,
    format_gff3_attributes,
    parse_attributes,
    split_attributes,
)
from oriongeno.genome_annotation import parse_gtf_attributes


def test_attribute_tokens_preserve_style_and_duplicate_values() -> None:
    text = 'gene_id "g1"; tag "first"; tag "second";'
    tokens = split_attributes(text)

    assert [token.separator for token in tokens] == [" ", " ", " "]
    assert attribute_values(tokens)["tag"] == ["first", "second"]
    assert parse_attributes(text)["tag"] == "second"
    assert format_attributes(tokens) == text


def test_shared_parser_handles_gff3_and_public_compatibility_api() -> None:
    text = "ID=t1;Parent=g1,g2;Name=alpha beta"

    assert parse_attributes(text) == {"ID": "t1", "Parent": "g1,g2", "Name": "alpha beta"}
    assert parse_gtf_attributes(text) == parse_attributes(text)
    assert format_gff3_attributes([("Name", "alpha beta"), ("Parent", ["g1", "g2"])]) == (
        "Name=alpha%20beta;Parent=g1,g2"
    )


def test_attribute_formatter_keeps_explicit_token_style() -> None:
    assert format_attributes([AttributeToken("ID", "t1", "=")]) == "ID=t1;"
