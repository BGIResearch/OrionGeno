#!/usr/bin/env python3
"""GTF annotation data structures and serialization helpers."""

import sys
import csv

GTF_GENE_ID = "gene_id"
GTF_TRANSCRIPT_ID = "transcript_id"
GTF_OUTPUT_FEATURE_MAP = {
    "5UTR": "five_prime_UTR",
    "3UTR": "three_prime_UTR",
}
GTF_OUTPUT_SKIP_FEATURES = {"intron"}
GTF_FRAME_FEATURES = {"CDS", "start_codon", "stop_codon"}


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


def parse_gtf_attributes(attributes):
    """Parse GTF/GFF-style attributes into a dictionary.

    The annotation object stores plain internal IDs, while GTF output needs a
    structured ninth column. This parser keeps both forms explicit and also
    accepts bare IDs as a caller-provided fallback.
    """
    parsed = {}
    for item in str(attributes).strip().rstrip(";").split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" in item and " " not in item.split("=", 1)[0]:
            key, value = item.split("=", 1)
        else:
            parts = item.split(None, 1)
            if len(parts) != 2:
                continue
            key, value = parts
        parsed[key.strip()] = value.strip().strip('"')
    return parsed


def get_gtf_attribute(attributes, key, default=""):
    return parse_gtf_attributes(attributes).get(key, default)


def format_gtf_attributes(gene_id="", transcript_id="", extra_attributes=None):
    parts = []
    if gene_id:
        parts.append(f'{GTF_GENE_ID} "{gene_id}";')
    if transcript_id:
        parts.append(f'{GTF_TRANSCRIPT_ID} "{transcript_id}";')
    if extra_attributes:
        parts.extend(str(attr).strip() for attr in extra_attributes if str(attr).strip())
    return " ".join(parts)


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
        self.gtf = []
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
        if not (line[0] == self.chr or line[6] == self.strand):
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
        self.find_introns()
        if not self.check_cds_exons():
            return False
        self.find_transcript()
        self.find_start_stop_codon()
        return True

    def check_cds_exons(self):
        """Return True when the transcript has CDS or exon features."""
        if 'CDS' not in self.transcript_lines.keys() and 'exon' not in self.transcript_lines.keys():
            sys.stderr.write('Skipping transcript {}, no CDS nor exons in {}\n'.format(self.id, self.id))
            return False
        return True

    def find_introns(self):
        """Infer intron rows from ordered CDS or exon rows."""
        if not 'intron' in self.transcript_lines.keys():
            self.transcript_lines.update({'intron' : []})
            key = ''
            if 'CDS' in self.transcript_lines.keys():
                key = 'CDS'
            elif 'exon' in self.transcript_lines.keys():
                key = 'exon'
            if key:
                exon_lst = []
                for line in self.transcript_lines[key]:
                    exon_lst.append(line)
                exon_lst = sorted(exon_lst, key=lambda e:e[0])
                for i in range(1, len(exon_lst)):
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
        if not 'transcript' in self.transcript_lines.keys():
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
        """Infer start_codon and stop_codon rows when possible."""

        if not 'start_codon' in self.transcript_lines.keys():
            self.transcript_lines.update({'start_codon' : []})
        if not 'stop_codon' in self.transcript_lines.keys():
            self.transcript_lines.update({'stop_codon' : []})


        key = ''
        if 'CDS' in self.transcript_lines.keys():
            key = 'CDS'
        elif 'exon' in self.transcript_lines.keys():
            key = 'exon'
        if key:
            self.transcript_lines[key].sort(key = lambda x : x[3])
            tx = self.transcript_lines[key][0]
            line1 = [self.chr, tx[1], '', tx[3], tx[3] + 2, \
            '.', self.strand, '0', format_gtf_attributes(self.gene_id, self.id)]
            tx = self.transcript_lines[key][-1]
            line2 = [self.chr, tx[1], '', tx[4] - 2, tx[4], \
            '.', self.strand, '0', format_gtf_attributes(self.gene_id, self.id)]

            fragmented_transcript = True
            if tx[6] == '+':
                line1[2] = 'start_codon'
                line2[2] = 'stop_codon'
                if self.transcript_lines[key][0][7] == 0:
                    fragmented_transcript = False
                start = line1
                stop = line2
            else:
                line1[2] = 'stop_codon'
                line2[2] = 'start_codon'
                if self.transcript_lines[key][-1][7] == 0:
                    fragmented_transcript = False
                stop = line1
                start = line2
            if not 'start_codon' in self.transcript_lines.keys() and not fragmented_transcript:
                if not fragmented_transcript:
                    self.add_line(start)
                else:
                    self.transcript_lines.update({'start_codon' : []})
            if not 'stop_codon' in self.transcript_lines.keys():
                self.add_line(stop)

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
        for k in self.transcript_lines.keys():
            for i, raw_line in enumerate(self.transcript_lines[k]):
                g = list(raw_line)
                if k == 'transcript':
                    tx_line  = g
                    tx_line[8] = format_gtf_attributes(self.gene_id, tx_id)
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
                elif not k in ['transcript', 'gene']:
                    g[8] = format_gtf_attributes(self.gene_id, tx_id)
                normalized = normalize_gtf_output_line(g)
                if normalized is not None:
                    gtf.append(normalized)

        if not 'exon' in self.transcript_lines.keys():
            for g in [line for line in gtf if line[2] == 'CDS']:
                exon = list(g)
                exon[2] = 'exon'
                exon[7] = "."
                exon[8] = format_gtf_attributes(self.gene_id, tx_id)
                normalized_exon = normalize_gtf_output_line(exon)
                if normalized_exon is not None:
                    gtf.append(normalized_exon)

        gtf = sorted(gtf, key=lambda g: (g[3],g[4]))
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
        if not gene_id in self.genes.keys():
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
        if not t_id in self.transcripts.keys():
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
            gene_line[8] = format_gtf_attributes(gene_id)
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
        old_gene_gtf = sorted(self.gene_gtf.values(), key=lambda g: (g[0],g[3],g[4]))
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
            for old_tx_id in old_genes[old_gene_id]:
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
        with open(out_path, 'w+') as file:
            out_writer = csv.writer(file, delimiter='\t', quotechar = "|", lineterminator = '\n')
            for line in self.get_gtf():
                out_writer.writerow(line)
