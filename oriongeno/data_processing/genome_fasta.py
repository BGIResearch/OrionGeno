import numpy as np
import gzip, bz2
import math

from .sequence_planning import adapted_core_chunk_size

class GenomeSequences:
    def __init__(
        self,
        fasta_file="",
        genome=None,
        np_file="",
        chunksize=20000,
        overlap=1000,
        min_seq_len=0,
        upper_only=True,
    ):
        """Load genome sequences and prepare chunking parameters.

        Arguments:
            fasta_file (str): Path to the FASTA file containing genome sequences.
            genome: Optional in-memory sequence record mapping.
            np_file (str): Path to the numpy file containing one-hot encoded sequences.
            chunksize (int): Size of each chunk for splitting sequences.
            overlap (int): Overlap size between consecutive chunks.
            min_seq_len (int): Minimum sequence length to keep.
            upper_only (bool): Use 5-channel uppercase encoding when True.
        """
        self.fasta_file = fasta_file
        self.genome = genome
        self.np_file = np_file
        self.chunksize = chunksize
        self.overlap = overlap
        self.min_seq_len = min_seq_len
        self.sequences = []
        self.sequence_names = []
        self.one_hot_encoded = None
        self.chunks_one_hot = None
        self.chunks_seq = None
        self.upper_only = upper_only
        if self.genome:
            self.extract_seqarray()
        elif self.fasta_file:
            self.read_fasta()
        else:
            self.load_np_array(self.np_file)

    def extract_seqarray(self):
        """Extract the sequence array from the genome object."""
        for name, seqrec in self.genome.items():
            if len(seqrec.seq) < self.min_seq_len:
                continue
            self.sequences.append(str(seqrec.seq) if not self.upper_only else str(seqrec.seq).upper())
            self.sequence_names.append(name)

    def read_fasta(self):
        """Read genome sequences from plain, gzip, or bzip2 FASTA files."""
        opener = open
        if self.fasta_file.endswith(".gz"):
            opener = gzip.open
        elif self.fasta_file.endswith(".bz2"):
            opener = bz2.open

        current_name = ""
        current_parts = []

        def flush_record():
            if not current_name:
                return
            sequence = "".join(current_parts)
            if len(sequence) >= self.min_seq_len:
                self.sequences.append(sequence if not self.upper_only else sequence.upper())
                self.sequence_names.append(current_name)

        with opener(self.fasta_file, "rt") as file:
            for line in file:
                if line.startswith(">"):
                    flush_record()
                    current_name = line[1:].strip()
                    current_parts = []
                else:
                    current_parts.append(line.strip())
        flush_record()

    def encode_sequences(self, seq=None):
        """Encode selected sequences and store them in ``self.one_hot_encoded``."""
        if not seq:
            seq = self.sequence_names

        self.one_hot_encoded = {}

        for s in seq:
            sequence = self.sequences[self.sequence_names.index(s)]
            if self.upper_only:
                table = np.zeros((256, 5), dtype=np.uint8)
            else:
                table = np.zeros((256, 9), dtype=np.uint8)
            table[:, 4] = 1
            if not self.upper_only:
                # Preserve lowercase bases as separate softmask channels.
                table[ord("A"), :] = [1, 0, 0, 0, 0, 0, 0, 0, 0]
                table[ord("C"), :] = [0, 1, 0, 0, 0, 0, 0, 0, 0]
                table[ord("G"), :] = [0, 0, 1, 0, 0, 0, 0, 0, 0]
                table[ord("T"), :] = [0, 0, 0, 1, 0, 0, 0, 0, 0]
                table[ord("a"), :] = [0, 0, 0, 0, 0, 1, 0, 0, 0]
                table[ord("c"), :] = [0, 0, 0, 0, 0, 0, 1, 0, 0]
                table[ord("g"), :] = [0, 0, 0, 0, 0, 0, 0, 1, 0]
                table[ord("t"), :] = [0, 0, 0, 0, 0, 0, 0, 0, 1]
            elif self.upper_only:
                table[ord('A'), :] = [1, 0, 0, 0, 0]
                table[ord('C'), :] = [0, 1, 0, 0, 0]
                table[ord('G'), :] = [0, 0, 1, 0, 0]
                table[ord('T'), :] = [0, 0, 0, 1, 0]
            # A byte lookup table is faster than per-character loops.
            int_seq = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
            self.one_hot_encoded[s] = table[int_seq]

    def reverse_complement(self, sequence):
        """Get the reverse complement of a DNA sequence.

        Arguments:
            sequence (str): The DNA sequence to reverse complement.

        Returns:
            str: The reverse complement of the input sequence.
        """
        complement = {
            "A": "T",
            "T": "A",
            "C": "G",
            "G": "C",
            "a": "t",
            "t": "a",
            "c": "g",
            "g": "c",
            "N": "N",
            "n": "n",
        }
        return "".join(complement.get(base, base) for base in reversed(sequence))

    def get_flat_chunks(
        self,
        sequence_names=None,
        strand="+",
        coords=False,
        pad=True,
        adapt_chunksize=False,
        parallel_factor=None,
        flank_size=0,
    ):
        """Get flattened chunks of a specific sequence by name.

        Arguments:
            sequence_names (list of str, optional): Names of sequences to extract chunks from.
                Defaults to all loaded sequences.
            strand (char): Strand direction ('+' for forward, '-' for reverse).
            flank_size (int): Bases kept as left/right context inside each model window.

        Returns:
            chunks_one_hot (np.array): Flattened chunks of the specified sequence.
            chunk_coords (list): List of central output coordinates if coords is True.
            model_chunksize (int): Model input size including flanks.
        """
        if flank_size < 0:
            raise ValueError("flank_size must be non-negative.")

        chunk_coords = None
        chunksize = self.chunksize

        if not sequence_names:
            sequence_names = self.sequence_names
        sequences_i = [self.one_hot_encoded[i] for i in sequence_names]

        if adapt_chunksize:
            max_len = max([len(seq) for seq in sequences_i])
            chunksize = adapted_core_chunk_size(
                max_len,
                chunk_size=self.chunksize,
                parallel_factor=parallel_factor,
                upper_only=self.upper_only,
                overlap=self.overlap,
            )

        core_chunksize = chunksize
        model_chunksize = core_chunksize + 2 * flank_size

        def unknown_padding(length, channels, dtype):
            padding = np.zeros((length, channels), dtype=dtype)
            padding[:, 4] = 1
            return padding

        def slice_with_padding(sequence, start, end):
            channels = sequence.shape[1]
            left_pad = max(0, -start)
            right_pad = max(0, end - len(sequence))
            clipped_start = max(0, start)
            clipped_end = min(len(sequence), end)
            parts = []
            if left_pad:
                parts.append(unknown_padding(left_pad, channels, sequence.dtype))
            if clipped_end > clipped_start:
                parts.append(sequence[clipped_start:clipped_end, :])
            if right_pad:
                parts.append(unknown_padding(right_pad, channels, sequence.dtype))
            if not parts:
                return unknown_padding(end - start, channels, sequence.dtype)
            chunk = np.concatenate(parts, axis=0)
            expected_len = end - start
            if chunk.shape[0] != expected_len:
                raise ValueError(
                    f"Internal chunk length mismatch: got {chunk.shape[0]}, expected {expected_len}."
                )
            return chunk

        chunks_one_hot = []
        if coords:
            chunk_coords = []
        for seq_name, sequence in zip(sequence_names, sequences_i):
            if pad:
                num_chunks = max(1, int(math.ceil(len(sequence) / core_chunksize)))
            else:
                num_chunks = len(sequence) // core_chunksize

            for i in range(num_chunks):
                core_start = i * core_chunksize
                core_end = core_start + core_chunksize
                valid_core_end = min(core_end, len(sequence))
                window_start = core_start - flank_size
                window_end = core_end + flank_size
                chunks_one_hot.append(slice_with_padding(sequence, window_start, window_end))
                if coords:
                    chunk_coords.append(
                        [seq_name, strand, core_start + 1, valid_core_end]
                    )

        chunks_one_hot = np.stack(chunks_one_hot, axis=0)
        if strand == "-":
            if not self.upper_only:
                # Reverse-complement 9-channel input with softmask tracks.
                chunks_one_hot = chunks_one_hot[::-1, ::-1, [3, 2, 1, 0, 4, 8, 7, 6, 5]]
            else:
                # Reverse-complement 5-channel nucleotide input.
                chunks_one_hot = chunks_one_hot[::-1, ::-1, [3, 2, 1, 0, 4]]

            if chunk_coords:
                chunk_coords.reverse()

        return chunks_one_hot, chunk_coords, model_chunksize
