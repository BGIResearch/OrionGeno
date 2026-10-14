"""Annotation data structures and GTF/GFF3 serialization helpers."""

import sys
from urllib.parse import unquote

from .annotation_io import (
    GTF_GENE_ID,
    GTF_TRANSCRIPT_ID,
    format_gff3_attributes,
    format_parsed_gff3_attributes,
    format_gtf_attributes,
    open_annotation_text,
    parse_attributes as parse_gtf_attributes,
    split_attributes,
)

GTF_OUTPUT_FEATURE_MAP = {
    "5UTR": "five_prime_UTR",
    "3UTR": "three_prime_UTR",
}
GTF_OUTPUT_SKIP_FEATURES = {"intron"}
GTF_FRAME_FEATURES = {"CDS", "start_codon", "stop_codon"}
GFF3_TRANSCRIPT_FEATURE = "mRNA"
GFF3_CHILD_FEATURE_SUFFIX = {
    "exon": "exon",
    "CDS": "cds",
    "five_prime_UTR": "five_prime_UTR",
    "three_prime_UTR": "three_prime_UTR",
    "start_codon": "start_codon",
    "stop_codon": "stop_codon",
}


def normalize_gtf_output_line(line):
    """Return a standards-friendly GTF row, or None when the row is skipped."""
    normalized = list(line)
    feature = str(normalized[2])
    if feature in GTF_OUTPUT_SKIP_FEATURES:
        return None
    normalized[2] = GTF_OUTPUT_FEATURE_MAP.get(feature, feature)
    if normalized[2] not in GTF_FRAME_FEATURES:
        normalized[7] = "."
    return normalized


def get_gtf_attribute(attributes, key, default=""):
    return parse_gtf_attributes(attributes).get(key, default)


def _gff3_child_id(parent_id, feature, index):
    suffix = GFF3_CHILD_FEATURE_SUFFIX.get(feature, str(feature).replace(" ", "_"))
    return f"{parent_id}.{suffix}{index}"


def gtf_rows_to_gff3_rows(rows):
    """Yield GFF3 rows converted from OrionGeno-normalized GTF rows."""
    child_counts = {}

    for row in rows:
        fields = [str(item) for item in row]
        if len(fields) != 9:
            continue

        feature = fields[2]
        parsed = parse_gtf_attributes(fields[8])
        tokens = split_attributes(fields[8])
        if tokens and tokens[0].separator == "=":
            fields[8] = format_parsed_gff3_attributes(parsed.items())
            yield fields
            continue
        attrs = []

        if feature == "gene":
            gene_id = (
                parsed.pop(GTF_GENE_ID, "")
                or parsed.pop("ID", "")
                or fields[8].strip()
            )
            parsed.pop(GTF_TRANSCRIPT_ID, None)
            attrs.append(("ID", gene_id))
        elif feature == "transcript":
            gene_id = parsed.pop(GTF_GENE_ID, "")
            transcript_id = parsed.pop(GTF_TRANSCRIPT_ID, "") or parsed.pop("ID", "")
            parsed.pop("Parent", None)
            fields[2] = GFF3_TRANSCRIPT_FEATURE
            attrs.append(("ID", transcript_id))
            if gene_id:
                attrs.append(("Parent", gene_id))
        else:
            gene_id = parsed.pop(GTF_GENE_ID, "")
            transcript_id = parsed.pop(GTF_TRANSCRIPT_ID, "")
            parent_id = transcript_id or gene_id
            repeat_id = parsed.pop("repeat_id", "")
            existing_id = parsed.pop("ID", "")
            existing_parent = parsed.pop("Parent", "")

            if parent_id:
                count_key = (parent_id, feature)
                child_counts[count_key] = child_counts.get(count_key, 0) + 1
                attrs.append(("ID", _gff3_child_id(parent_id, feature, child_counts[count_key])))
                attrs.append(("Parent", parent_id))
            elif repeat_id:
                attrs.append(("ID", repeat_id))
            else:
                if existing_id:
                    attrs.append(("ID", existing_id))
                if existing_parent:
                    attrs.append(("Parent", existing_parent))

        for key, value in parsed.items():
            if key in {GTF_GENE_ID, GTF_TRANSCRIPT_ID}:
                continue
            attrs.append((key, value))

        if fields[2] != "CDS":
            fields[7] = "."
        fields[8] = format_gff3_attributes(attrs)
        yield fields


def gff3_rows_to_gtf_rows(rows):
    """Convert predictor gene/mRNA/child rows, retaining GTF input as GTF."""
    transcript_genes = {}
    converted = []
    codon_groups = {}
    for row in rows:
        fields = [str(item) for item in row]
        parsed = parse_gtf_attributes(fields[8])
        tokens = split_attributes(fields[8])
        if not tokens or tokens[0].separator != "=":
            converted.append(fields)
            continue
        feature_id = unquote(parsed.pop("ID", ""))
        parents = [unquote(parent) for parent in parsed.pop("Parent", "").split(",") if parent]
        if len(parents) > 1:
            raise ValueError("Prediction GTF export requires a single parent per feature")
        parent = parents[0] if parents else ""
        feature = fields[2]
        if feature == "gene":
            gene_id, transcript_id = feature_id, ""
        elif feature in {"mRNA", "transcript"}:
            gene_id, transcript_id = parent, feature_id
            transcript_genes[transcript_id] = gene_id
            fields[2] = "transcript"
        else:
            transcript_id = parent
            gene_id = transcript_genes.get(parent, "")
        extras = [f'{key} "{unquote(value)}";' for key, value in parsed.items()
                  if key not in {GTF_GENE_ID, GTF_TRANSCRIPT_ID}]
        fields[8] = format_gtf_attributes(gene_id, transcript_id, extras)
        if feature in {"start_codon", "stop_codon"}:
            codon_groups.setdefault((transcript_id, feature), []).append(fields)
        converted.append(fields)
    # Recover phases for codons split across exons, in transcript direction.
    for codon in codon_groups.values():
        consumed = 0
        for fields in sorted(codon, key=lambda line: int(line[3]), reverse=codon[0][6] == "-"):
            fields[7] = str((-consumed) % 3)
            consumed += int(fields[4]) - int(fields[3]) + 1
    yield from converted


def write_gff3_rows(rows, out_path):
    with open_annotation_text(out_path, "w") as file_obj:
        file_obj.write("##gff-version 3\n")
        for fields in gtf_rows_to_gff3_rows(rows):
            file_obj.write("\t".join(map(str, fields)) + "\n")


class NotGtfFormat(Exception):
    pass

class Transcript:
    """Store one transcript and derive missing GTF feature rows."""

    def __init__(self, id, gene_id, chr, source_anno, strand):
        """Create a transcript container.

        Args:
            id (str): Transcript ID.
            gene_id (str): Gene ID.
            chr (str): Chromosome or sequence name.
            source_anno (str): Annotation source ID.
            strand (str): Transcript strand ("+" or "-").
        """
        self.id = id
        self.chr = chr
        self.gene_id = gene_id
        # Feature type -> list of parsed GTF rows.
        self.transcript_lines = {}
        self.source_anno = source_anno
        self.start = -1
        self.end = -1
        self.cds_len = -1
        self.cds_coords = {}
        self.strand = strand
        self.source_method = ''

    def add_line(self, line):
        """Add a parsed GTF row to this transcript.

        Args:
            line (list): Nine-column GTF row represented as a mutable list.
        """
        if line[0] != self.chr or line[6] != self.strand:
            raise NotGtfFormat('File is not in gtf format. ' \
                + 'Error in line {}\n'.format('\t'.join(map(str, line)))
                + 'Transcript ID is not unique')

        if line[2] not in self.transcript_lines.keys():
            self.transcript_lines.update({line[2] : []})

        self.source_method = line[1]

        line[3] = int(line[3])
        line[4] = int(line[4])
        if self.start < 0 or line[3] < self.start:
            self.start = line[3]
        if self.end < 0 or line[4] > self.end:
            self.end = line[4]
        if self.gene_id == '' and not line[2] == 'transcript':
            self.gene_id = get_gtf_attribute(line[8], GTF_GENE_ID, self.gene_id)
        self.transcript_lines[line[2]].append(line)

    def get_type_coords(self, type, frame=True):
        """Return coordinates for the requested feature type.

        Returns:
            Either a phase-keyed coordinate dictionary or a flat coordinate list.
        """
        if frame:
            coords = {'0' : [], '1' : [], '2' : [], '.' : []}
        else:
            coords = []
        if type == 'CDS' and type not in self.transcript_lines.keys():
            type = 'exon'
        if type not in self.transcript_lines.keys():
            return coords
        for line in self.transcript_lines[type]:
            if frame:
                coords[line[7]].append([line[3], line[4]])
            else:
                coords.append([line[3], line[4]])
        if frame:
            for k in coords.keys():
                coords[k].sort(key=lambda c: (c[0],c[1]))
            if type == 'CDS':
                coords['0'] += coords['.']
                del coords['.']
        else:
            coords.sort(key=lambda c: (c[0],c[1]))
        return coords

    def get_cds_len(self):
        cds = self.get_type_coords('CDS', False)
        return sum([c[1] - c[0] + 1 for c in cds])

    def add_missing_lines(self):
        """Add missing transcript, intron, exon, and codon rows.

        Returns:
            bool: False if no CDS or exon rows were found, otherwise True.
        """
        self.find_exons()
        if not self.check_cds_exons():
            return False
        self.find_introns()
        self.find_transcript()
        self.find_start_stop_codon()
        return True

    def _inferred_exon_lines(self):
        """Union adjacent CDS/UTR intervals into complete transcript exons."""
        pieces = sorted(
            (
                list(line)
                for feature in ("CDS", "5UTR", "3UTR", "five_prime_UTR", "three_prime_UTR")
                for line in self.transcript_lines.get(feature, [])
            ),
            key=lambda line: (line[3], line[4]),
        )
        exons = []
        for line in pieces:
            if exons and line[3] <= exons[-1][4] + 1:
                exons[-1][4] = max(exons[-1][4], line[4])
                continue
            line[2] = "exon"
            line[7] = "."
            line[8] = format_gtf_attributes(self.gene_id, self.id)
            exons.append(line)
        return exons

    def find_exons(self):
        """Fill absent exons from CDS and UTR without changing either feature."""
        if not self.transcript_lines.get("exon"):
            for line in self._inferred_exon_lines():
                self.add_line(line)

    def check_cds_exons(self):
        """Return True when the transcript has CDS or exon features."""
        if 'CDS' not in self.transcript_lines.keys() and 'exon' not in self.transcript_lines.keys():
            sys.stderr.write('Skipping transcript {}, no CDS nor exons in {}\n'.format(self.id, self.id))
            return False
        return True

    def find_introns(self):
        """Infer intron rows from ordered CDS or exon rows."""
        if 'intron' not in self.transcript_lines:
            self.transcript_lines.update({'intron' : []})
            key = ''
            if self.transcript_lines.get('exon'):
                key = 'exon'
            elif self.transcript_lines.get('CDS'):
                key = 'CDS'
            if key:
                exon_lst = []
                for line in self.transcript_lines[key]:
                    exon_lst.append(line)
                exon_lst = sorted(exon_lst, key=lambda e:e[3])
                for i in range(1, len(exon_lst)):
                    if exon_lst[i][3] <= exon_lst[i-1][4] + 1:
                        continue
                    intron = []
                    intron += exon_lst[i][0:2]
                    intron.append('intron')
                    intron.append(exon_lst[i-1][4] + 1)
                    intron.append(exon_lst[i][3] - 1)
                    intron += exon_lst[i][5:8]
                    intron.append(format_gtf_attributes(self.gene_id, self.id))
                    self.transcript_lines['intron'].append(intron)

    def find_transcript(self):
        """Infer a transcript row from the feature span."""
        if 'transcript' not in self.transcript_lines:
            if not any(self.transcript_lines.values()):
                return
            for k in self.transcript_lines.keys():
                for line in self.transcript_lines[k]:
                    if line[3] < self.start or self.start < 0:
                        self.start = line[3]
                    if line[4] > self.end:
                        self.end = line[4]
            tx_line = [self.chr, line[1], 'transcript', self.start, self.end, \
            '.', line[6], '.', format_gtf_attributes(self.gene_id, self.id)]
            self.add_line(tx_line)

    def find_start_stop_codon(self):
        """Infer terminal codon intervals along the spliced CDS, on either strand.

        These positional features are not an independent sequence-level QC.
        Incomplete codons and exon-only models do not acquire codon rows.
        """
        cds = sorted(
            self.transcript_lines.get("CDS", []),
            key=lambda line: (line[3], line[4]),
            reverse=self.strand == "-",
        )
        if not cds:
            return
        total = sum(line[4] - line[3] + 1 for line in cds)
        if total < 3:
            return
        first_phase = str(cds[0][7])
        codons = []
        if first_phase == "0":
            codons.append(("start_codon", 0, 3))
        if first_phase in {"0", "1", "2"} and (total - int(first_phase)) % 3 == 0:
            codons.append(("stop_codon", total - 3, total))
        for feature, begin, end in codons:
            if self.transcript_lines.get(feature):
                continue
            offset = 0
            consumed = 0
            for segment in cds:
                length = segment[4] - segment[3] + 1
                left, right = max(begin, offset), min(end, offset + length)
                if left < right:
                    line = list(segment)
                    line[2], line[5], line[7] = feature, ".", str((-consumed) % 3)
                    if self.strand == "-":
                        line[3], line[4] = segment[4] - (right - offset) + 1, segment[4] - (left - offset)
                    else:
                        line[3], line[4] = segment[3] + left - offset, segment[3] + right - offset - 1
                    line[8] = format_gtf_attributes(self.gene_id, self.id)
                    self.add_line(line)
                    consumed += right - left
                offset += length

    def redo_phase(self):
        if 'CDS' in self.transcript_lines:
            self.transcript_lines['CDS'] = sorted(self.transcript_lines['CDS'],
                                                key=lambda x: x[3], reverse=self.strand=='-')
            phase = 0
            for line in self.transcript_lines['CDS']:
                line[7] = phase
                phase = (3 - (line[4] - line[3] + 1 - phase)%3)%3

    def check_splits(self):
        for k in self.transcript_lines.keys():
            if not self.transcript_lines[k]:
                continue
            self.transcript_lines[k] = sorted(self.transcript_lines[k],
                                              key=lambda x: x[3])
            new_list = [self.transcript_lines[k][0]]
            for i in range(1, len(self.transcript_lines[k])):
                if new_list[-1][4] == self.transcript_lines[k][i][3]-1:
                    new_list[-1][4] = self.transcript_lines[k][i][4]
                else:
                    new_list.append(self.transcript_lines[k][i])
            self.transcript_lines[k] = new_list

    def get_gtf(self, prefix=''):
        """Return GTF rows for this transcript.

        Returns:
            list: Parsed GTF rows ready for tab-delimited writing.
        """
        gtf = []
        if prefix:
            prefix += '.'
        tx_line = []
        tx_id = prefix + self.id
        coding = bool(self.transcript_lines.get('CDS'))
        for k in self.transcript_lines.keys():
            lines = sorted(self.transcript_lines[k], key=lambda line: (line[3], line[4]))
            for i, raw_line in enumerate(lines):
                g = list(raw_line)
                if k == 'transcript':
                    tx_line  = g
                    tx_line[8] = format_gtf_attributes(
                        self.gene_id, tx_id,
                        ['transcript_biotype "protein_coding";'] if coding else None,
                    )
                    continue
                elif k == 'CDS':
                    cds_type = 'internal'
                    if len(self.transcript_lines[k]) == 1:
                        cds_type = 'single'
                    elif (i == 0 and self.strand == '+') or (i == len(self.transcript_lines[k]) - 1 and self.strand == '-'):
                        cds_type = 'initial'
                    elif (i == len(self.transcript_lines[k]) - 1 and self.strand == '+') or (i == 0 and self.strand == '-'):
                        cds_type = 'terminal'
                    g[8] = format_gtf_attributes(
                        self.gene_id,
                        tx_id,
                        [f'cds_type "{cds_type}";'],
                    )
                elif k not in ['transcript', 'gene']:
                    g[8] = format_gtf_attributes(self.gene_id, tx_id)
                normalized = normalize_gtf_output_line(g)
                if normalized is not None:
                    gtf.append(normalized)

        if not self.transcript_lines.get('exon'):
            for exon in self._inferred_exon_lines():
                exon[8] = format_gtf_attributes(self.gene_id, tx_id)
                normalized_exon = normalize_gtf_output_line(exon)
                if normalized_exon is not None:
                    gtf.append(normalized_exon)

        feature_order = {"exon": 0, "five_prime_UTR": 1, "CDS": 2,
                         "three_prime_UTR": 3, "start_codon": 4, "stop_codon": 5}
        gtf = sorted(gtf, key=lambda g: (g[3], feature_order.get(g[2], 6), g[4], g[2]))
        if tx_line:
            normalized_tx = normalize_gtf_output_line(tx_line)
            if normalized_tx is not None:
                gtf = [normalized_tx] + gtf
        return gtf

class Anno:
    """Represent a genome annotation as genes and transcripts."""

    def __init__(self, path, id):
        """Create an annotation container.

        Args:
            path (str): Input or output GTF path.
            id (str): Annotation source ID.
        """
        self.id = id
        self.genes = {'None' : []}
        self.gene_gtf = {}
        self.transcripts = {}
        self.path = path
        self.translation_tab = []

    def norm_tx_format(self):
        """Fill missing transcript features and drop transcripts without CDS/exons."""
        tx_no_cds = []
        for k in self.transcripts.keys():
            if not self.transcripts[k].add_missing_lines():
                tx_no_cds.append(k)
        for k in tx_no_cds:
            del self.transcripts[k]

    def genes_update(self, gene_id, transcript_id=''):
        """Update the gene-to-transcript index.

        Args:
            gene_id (str): Gene ID.
            transcript_id (str): Transcript ID.
        """
        if gene_id not in self.genes:
            self.genes.update({ gene_id : []})
        if transcript_id and transcript_id not in self.genes[gene_id]:
            self.genes[gene_id].append(transcript_id)
        if transcript_id in self.genes['None'] and not gene_id == 'None':
            self.genes['None'].remove(transcript_id)
            self.transcripts[transcript_id].gene_id = gene_id

    def transcript_update(self, t_id, g_id, chr, strand):
        """Create a transcript entry if it is not already present.

        Args:
            t_id (str): Transcript ID.
            g_id (str): Gene ID.
            chr (str): Chromosome or sequence name.
            strand (str): Transcript strand ("+" or "-").
        """
        if t_id not in self.transcripts:
            self.transcripts.update({ t_id : Transcript(t_id, g_id, chr, self.id, strand)})

    def find_genes(self):
        """Group transcripts by gene and derive gene-level GTF rows."""
        self.gene_gtf = {}
        self.genes = {}
        for tx in self.transcripts.values():
            if tx.gene_id in self.genes.keys():
                if not (tx.chr == self.gene_gtf[tx.gene_id][0] and \
                    tx.strand == self.gene_gtf[tx.gene_id][6]):
                    tx.gene_id = tx.gene_id + '.' + tx.chr + '.' + tx.strand
                else:
                    self.genes[tx.gene_id].append(tx.id)
                    self.gene_gtf[tx.gene_id][3] = min(self.gene_gtf[tx.gene_id][3], \
                        tx.start)
                    self.gene_gtf[tx.gene_id][4] = max(self.gene_gtf[tx.gene_id][4], \
                        tx.end)
                    continue
            self.genes.update({tx.gene_id : [tx.id]})
            self.gene_gtf.update({tx.gene_id : [tx.chr, tx.source_method, 'gene', \
                tx.start, tx.end, '.', tx.strand, '.', tx.gene_id]})

    def get_gtf(self):
        """Return the annotation as sorted GTF rows.

        Returns:
            list: GTF rows represented as lists.
        """
        gtf = []
        gene_gtf = sorted(self.gene_gtf.values(), key=lambda g: (g[0],g[3],g[4]))
        for gene in gene_gtf:
            gene_id = gene[8]
            gene_line = list(gene)
            coding = any(self.transcripts[tx].transcript_lines.get('CDS') for tx in self.genes[gene_id])
            gene_line[8] = format_gtf_attributes(
                gene_id, extra_attributes=['gene_biotype "protein_coding";'] if coding else None,
            )
            normalized_gene = normalize_gtf_output_line(gene_line)
            if normalized_gene is not None:
                gtf.append(normalized_gene)
            for tx_id in self.genes[gene_id]:
                gtf += self.transcripts[tx_id].get_gtf()
        return gtf

    def add_transcripts(self, txs, id_prefix=''):
        """Add transcript objects to the annotation.

        Args:
            txs (dict): Mapping of transcript IDs to Transcript objects.
            id_prefix (str): Optional prefix applied to transcript IDs.
        """
        if not id_prefix:
            self.transcripts.update(txs)
        else:
            for tx in txs.values():
                tx.id = id_prefix + tx.id
                self.transcripts.update({tx.id : tx})

    def rename_tx_ids(self, prefix=''):
        """Rename genes and transcripts in genomic order.

        Args:
            prefix (str): Optional prefix added before each new gene and transcript ID.

        Returns:
            list: Translation table mapping new transcript IDs to old transcript IDs.
        """
        self.translation_tab = []
        gene_numb = 1
        old_gene_gtf = sorted(self.gene_gtf.values(), key=lambda g: (g[0],g[3],g[4],g[6]))
        self.gene_gtf = {}
        old_genes = self.genes
        self.genes = {}
        old_txs = self.transcripts
        self.transcripts = {}
        if prefix:
            prefix += '_'
        for gene in old_gene_gtf:
            tx_numb = 1
            old_gene_id = gene[8]
            new_gene_id = "{}g{}".format(prefix, gene_numb)
            gene[8] = new_gene_id
            self.genes.update({new_gene_id : []})
            self.gene_gtf.update({new_gene_id : gene})
            for old_tx_id in sorted(
                old_genes[old_gene_id],
                key=lambda tx_id: (old_txs[tx_id].start, old_txs[tx_id].end,
                                   old_txs[tx_id].get_type_coords('CDS', False)),
            ):
                new_tx_id = "{}g{}.t{}".format(prefix, gene_numb, tx_numb)
                self.transcripts.update({new_tx_id : old_txs[old_tx_id]})
                self.transcripts[new_tx_id].id = new_tx_id
                self.transcripts[new_tx_id].gene_id = new_gene_id
                self.genes[new_gene_id].append(new_tx_id)
                tx_numb +=1
                self.translation_tab.append([new_tx_id, old_tx_id])
            gene_numb += 1
        return self.translation_tab

    def write_anno(self, out_path):
        """Write the annotation to a tab-delimited GTF file.

        Args:
            out_path (str): Output GTF path.
        """
        with open_annotation_text(out_path, 'w') as file:
            for line in self.get_gtf():
                file.write('\t'.join(map(str, line)) + '\n')

    def write_gff3(self, out_path):
        """Write the annotation to a tab-delimited GFF3 file."""
        write_gff3_rows(self.get_gtf(), out_path)
