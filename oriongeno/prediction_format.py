"""Prediction GFF3 serialization with consistent gene and transcript identifiers.

Only the representation changes: exon/CDS coordinates and CDS phase are retained.
The OrionGeno source and OG namespace identify sequence-predicted models.
"""

from collections import defaultdict
from itertools import chain
import os
from urllib.parse import unquote

from Bio.Seq import Seq

from .annotation_io import format_gff3_attributes, open_annotation_text, parse_attributes, split_attributes
from .genome_annotation import gff3_rows_to_gtf_rows, gtf_rows_to_gff3_rows


PREDICTION_GENE_PREFIX = "OG"
_UTR_FEATURES = {"5UTR", "3UTR", "five_prime_UTR", "three_prime_UTR", "UTR"}
_TRANSCRIPT_ATTRIBUTES = {
    "ID", "Parent", "gene_id", "transcript_id", "geneID", "gene_biotype",
    "transcript_biotype", "StartCodon", "StopCodon", "Num_exons",
}


def _terminal_codons(cds, genome):
    if not cds or genome is None:
        return None, None
    ordered = sorted(cds, key=lambda row: int(row[3]), reverse=cds[0][6] == "-")
    parts = []
    for row in ordered:
        record = genome[row[0]]
        sequence = record.seq if hasattr(record, "seq") else record
        part = str(sequence[int(row[3]) - 1:int(row[4])]).upper()
        parts.append(str(Seq(part).reverse_complement()) if row[6] == "-" else part)
    sequence = "".join(parts)
    return (sequence[:3], sequence[-3:]) if len(sequence) >= 3 else (".", ".")


def _inferred_exons(features):
    exons = []
    for original in sorted(features, key=lambda row: (int(row[3]), int(row[4]))):
        row = list(original)
        if exons and int(row[3]) <= int(exons[-1][4]) + 1:
            exons[-1][4] = str(max(int(exons[-1][4]), int(row[4])))
        else:
            row[2], row[7] = "exon", "."
            exons.append(row)
    return exons


def prediction_gff3_rows(rows, genome=None, gene_prefix=PREDICTION_GENE_PREFIX):
    """Yield gene, mRNA, all exons, then all CDS rows for each predicted model.

    Input can be the internal GTF rows, older public GFF3, or this same schema.
    Existing codon attributes survive conversion without a genome; new unknown
    codons are written as '.', never guessed from phase or completeness.
    """
    genes, transcripts = {}, {}
    children = defaultdict(list)
    rows = iter(rows)
    first = next(rows, None)
    if first is None:
        return
    tokens = split_attributes(str(first[8]))
    input_rows = chain([first], rows)
    # GFF3 shards already carry their hierarchy. Only legacy GTF input needs
    # adaptation; the normal prediction path does not enter this reader.
    if not tokens or tokens[0].separator != "=":
        input_rows = gtf_rows_to_gff3_rows(input_rows)
    for row in input_rows:
        row = [str(value) for value in row]
        attrs = parse_attributes(row[8])
        identifier = attrs.get("ID", "")
        if row[2] == "gene":
            genes[identifier] = row
        elif row[2] in {"mRNA", "transcript"}:
            parents = attrs.get("Parent", "").split(",")
            if not identifier or len(parents) != 1 or not parents[0]:
                raise ValueError("Prediction transcripts require an ID and one gene Parent")
            if identifier in transcripts:
                raise ValueError(f"Duplicate prediction transcript ID: {identifier}")
            transcripts[identifier] = (row, attrs, parents[0])
        elif row[2] in {"exon", "CDS"} | _UTR_FEATURES:
            for parent in attrs.get("Parent", "").split(","):
                if not parent:
                    raise ValueError(f"Prediction {row[2]} requires a transcript Parent")
                children[parent].append(row)
    yield from _render_prediction_models(genes, transcripts, children, genome, gene_prefix)


def annotation_prediction_rows(annotation, genome=None, gene_prefix=PREDICTION_GENE_PREFIX):
    """Serialize native annotation objects directly, without GTF rows or parsing.

    Historical Transcript containers store feature columns in lists. Only their
    first eight columns are used here; identifiers come from object fields.
    """
    genes, transcripts = {}, {}
    children = defaultdict(list)
    for tx in annotation.transcripts.values():
        row = [tx.chr, "OrionGeno", "mRNA", str(tx.start), str(tx.end),
               ".", tx.strand, ".", "."]
        transcripts[tx.id] = (row, {}, tx.gene_id)
        if tx.gene_id in annotation.gene_gtf:
            genes[tx.gene_id] = [str(value) for value in annotation.gene_gtf[tx.gene_id][:8]] + ["."]
        for feature in {"exon", "CDS"} | _UTR_FEATURES:
            for segment in tx.transcript_lines.get(feature, []):
                children[tx.id].append([str(value) for value in segment[:8]] + ["."])
    yield from _render_prediction_models(genes, transcripts, children, genome, gene_prefix)


def _render_prediction_models(genes, transcripts, children, genome, gene_prefix):
    """Shared GFF3 renderer for native models and imported prediction files."""
    orphaned = children.keys() - transcripts.keys()
    if orphaned:
        raise ValueError(f"Prediction features have unknown transcript parents: {sorted(orphaned)[:5]}")

    gene_transcripts = defaultdict(list)
    for identifier, (row, attrs, gene_id) in transcripts.items():
        gene_transcripts[gene_id].append(identifier)
        if gene_id not in genes:
            gene = list(row)
            gene[2] = "gene"
            genes[gene_id] = gene
    def gene_key(gene_id):
        row = genes[gene_id]
        return row[0], int(row[3]), int(row[4]), row[6]

    for number, old_gene in enumerate(sorted(gene_transcripts, key=gene_key), 1):
        gene_id = f"{gene_prefix}_{number:08d}"
        tx_ids = sorted(gene_transcripts[old_gene], key=lambda key: (
            int(transcripts[key][0][3]), int(transcripts[key][0][4]),
            sorted((int(row[3]), int(row[4])) for row in children[key] if row[2] == "CDS"),
        ))
        coding_gene = any(row[2] == "CDS" for key in tx_ids for row in children[key])
        biotype = "protein_coding" if coding_gene else "unclassified"
        gene = list(genes[old_gene])
        gene[1], gene[7] = "OrionGeno", "."
        gene[3] = str(min(int(transcripts[key][0][3]) for key in tx_ids))
        gene[4] = str(max(int(transcripts[key][0][4]) for key in tx_ids))
        gene[8] = format_gff3_attributes([
            ("ID", gene_id), ("geneID", gene_id), ("gene_biotype", biotype),
        ])
        yield gene
        for tx_number, old_tx in enumerate(tx_ids, 1):
            tx_id = f"{gene_id}-mRNA-{tx_number}"
            original, attrs, _ = transcripts[old_tx]
            features = children[old_tx]
            cds = sorted((row for row in features if row[2] == "CDS"),
                         key=lambda row: (int(row[3]), int(row[4])))
            exons = sorted((row for row in features if row[2] == "exon"),
                           key=lambda row: (int(row[3]), int(row[4])))
            if not exons:
                exons = _inferred_exons(features)
            start_codon, stop_codon = _terminal_codons(cds, genome)
            tx = list(original)
            tx[1], tx[2], tx[7] = "OrionGeno", "mRNA" if cds else "transcript", "."
            tx_attrs = [("ID", tx_id), ("Parent", gene_id)]
            if cds:
                tx_attrs.extend([
                    ("StartCodon", start_codon if start_codon is not None else unquote(attrs.get("StartCodon", "."))),
                    ("StopCodon", stop_codon if stop_codon is not None else unquote(attrs.get("StopCodon", "."))),
                ])
            tx_attrs.extend([
                ("Num_exons", len(exons)), ("geneID", gene_id),
                ("gene_biotype", "protein_coding" if cds else "unclassified"),
            ])
            tx_attrs.extend((key, unquote(value)) for key, value in attrs.items()
                            if key not in _TRANSCRIPT_ATTRIBUTES)
            tx[8] = format_gff3_attributes(tx_attrs)
            yield tx
            for feature, segments in (("exon", exons), ("CDS", cds)):
                for segment_number, original_segment in enumerate(segments, 1):
                    segment = list(original_segment)
                    segment[1], segment[2] = "OrionGeno", feature
                    if feature != "CDS":
                        segment[7] = "."
                    segment[8] = format_gff3_attributes([
                        ("Parent", tx_id), ("ID", f"{tx_id}:{feature}:{segment_number}"),
                    ])
                    yield segment


def write_prediction_rows(rows, output_path, genome=None, gene_prefix=PREDICTION_GENE_PREFIX):
    """Write imported prediction rows using the shared schema."""
    normalized = prediction_gff3_rows(rows, genome=genome, gene_prefix=gene_prefix)
    _write_formatted_rows(normalized, output_path)


def write_prediction_annotation(annotation, output_path, genome=None, gene_prefix=PREDICTION_GENE_PREFIX):
    """Write native prediction objects directly to the requested output format."""
    normalized = annotation_prediction_rows(annotation, genome=genome, gene_prefix=gene_prefix)
    _write_formatted_rows(normalized, output_path)


def _write_formatted_rows(normalized, output_path):
    path = os.fspath(output_path).lower()
    if path.endswith(".gz"):
        path = path[:-3]
    is_gff = os.path.splitext(path)[1] in {".gff", ".gff3"}
    if not is_gff:
        normalized = gff3_rows_to_gtf_rows(normalized)
    with open_annotation_text(output_path, "w") as handle:
        if is_gff:
            handle.write("##gff-version 3\n")
        for row in normalized:
            handle.write("\t".join(row) + "\n")


def normalize_prediction_file(input_path, output_path, genome=None):
    """Reformat an existing prediction, including old g1/g1.t1-style exports."""
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        raise ValueError("Use a separate destination when normalizing a prediction")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    def rows():
        with open_annotation_text(input_path) as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip() or line.startswith("#"):
                    continue
                fields = line.rstrip("\r\n").split("\t")
                if len(fields) != 9:
                    raise ValueError(f"Expected nine annotation columns at {input_path}:{number}")
                yield fields
    write_prediction_rows(rows(), output_path, genome=genome)


def main(argv=None):
    import argparse
    from Bio import SeqIO
    from .fasta_io import open_fasta_text

    parser = argparse.ArgumentParser(description="Normalize an existing OrionGeno prediction as gene/transcript/CDS GFF3.")
    parser.add_argument("--input", required=True, help="Existing prediction GFF3/GTF, optionally .gz.")
    parser.add_argument("--output", required=True, help="New GFF3/GTF output, optionally .gz.")
    parser.add_argument("--genome", help="Genome FASTA for StartCodon/StopCodon attributes; existing values are otherwise retained.")
    args = parser.parse_args(argv)
    genome = None
    if args.genome:
        with open_fasta_text(args.genome) as handle:
            genome = {record.id: record for record in SeqIO.parse(handle, "fasta")}
    normalize_prediction_file(args.input, args.output, genome=genome)


if __name__ == "__main__":
    main()
