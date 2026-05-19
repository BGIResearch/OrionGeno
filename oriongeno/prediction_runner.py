"""OrionGeno prediction orchestration and parser."""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time

import numpy as np

from .cli_config import check_file_exists, env_bool, env_int, env_str, str_to_bool
from .constants import (
    DEFAULT_FRAGMENTED_RECORD_THRESHOLD,
    DEFAULT_MAX_FASTA_RECORDS,
    DEFAULT_PACK_SPACER_LEN,
    DEFAULT_PACK_TARGET_SIZE,
    SEQUENCE_GROUP_SIZE,
)
from .gtf_outputs import (
    RepeatGtfWriter,
    filter_and_write_outputs,
    repeat_output_path,
)
from .data_processing.fasta_io import load_genome, should_skip_by_fasta_record_count
from .data_processing.fragmented import prepare_inference_genome
from .data_processing.sequence_planning import (
    check_flank_size,
    check_parallel_factor,
    compute_parallel_factor,
    group_sequences,
)


ASSEMBLY_MODE = env_str("ORIONGENO_ASSEMBLY_MODE", "auto")
FRAGMENTED_RECORD_THRESHOLD = env_int(
    "ORIONGENO_FRAGMENTED_RECORD_THRESHOLD",
    DEFAULT_FRAGMENTED_RECORD_THRESHOLD,
)
PACK_SPACER_LEN = env_int("ORIONGENO_PACK_SPACER_LEN", DEFAULT_PACK_SPACER_LEN)
PACK_TARGET_SIZE = env_int("ORIONGENO_PACK_TARGET_SIZE", DEFAULT_PACK_TARGET_SIZE)
MIN_SEQ_LEN = 0
PREDICTION_STRANDS = ("+", "-")
USE_HMM = True
UPPER_ONLY = True
GROUP_TARGET_SIZE = env_int("ORIONGENO_SEQUENCE_GROUP_SIZE", SEQUENCE_GROUP_SIZE)


class PredictionContext:
    def __init__(
        self,
        *,
        checkpoint_path,
        seq_len,
        flank_size,
        model_seq_len,
        batch_size,
        strands,
        species_name,
        parallel_factor,
        output_gene,
        output_repeat,
    ):
        self.checkpoint_path = checkpoint_path
        self.seq_len = seq_len
        self.flank_size = flank_size
        self.model_seq_len = model_seq_len
        self.batch_size = batch_size
        self.strands = strands
        self.species_name = species_name
        self.parallel_factor = parallel_factor
        self.output_gene = output_gene
        self.output_repeat = output_repeat


def _trim_prediction_flanks(array, flank_size):
    if flank_size <= 0:
        return array
    if array.shape[1] <= 2 * flank_size:
        raise ValueError(
            f"Cannot trim flank-size={flank_size} from prediction length {array.shape[1]}."
        )
    return array[:, flank_size:-flank_size, ...]


def _resolve_parallel_factor(args, model_seq_len, seq_len):
    requested = int(getattr(args, "hmm_parallel_factor", 0) or 0)
    if requested > 0:
        return requested
    return compute_parallel_factor(model_seq_len, core_seq_len=seq_len)


def _resolve_checkpoint_path(args):
    checkpoint = getattr(args, "checkpoint", "")
    if not checkpoint:
        logging.error("--checkpoint or ORIONGENO_CHECKPOINT is required.")
        sys.exit(1)
    return os.path.abspath(checkpoint)


def _strand_batches(strands):
    strands = tuple(strands)
    if strands == ("+", "-"):
        return [strands]
    return [(strand,) for strand in strands]


def _predict_genome_records(
    genome,
    context,
    *,
    annotation,
    transcript_counter,
    repeat_writer,
    coordinate_mapper,
    log_prefix="OrionGeno",
):
    from .eval_model_class import PredictionGTF

    inference_seq_dict = {}

    predictor = PredictionGTF(
        model_path=context.checkpoint_path,
        seq_len=context.seq_len,
        batch_size=context.batch_size,
        hmm=USE_HMM and context.output_gene,
        temp_dir=None,
        num_hmm=1,
        hmm_factor=1,
        genome=genome,
        upper_only=UPPER_ONLY,
        species_name=context.species_name,
        strand="+",
        parallel_factor=context.parallel_factor,
    )

    genome_fasta = predictor.init_fasta(
        chunk_len=context.seq_len,
        min_seq_len=MIN_SEQ_LEN,
    )
    inference_seq_dict.update(
        {
            seq_name: len(sequence)
            for seq_name, sequence in zip(
                genome_fasta.sequence_names,
                genome_fasta.sequences,
            )
        }
    )
    all_seq_names = list(genome_fasta.sequence_names)
    all_seq_lens = [len(sequence) for sequence in genome_fasta.sequences]
    logging.info(
        "%s loaded %s inference sequences from %s input sequences.",
        log_prefix,
        len(all_seq_names),
        len(genome),
    )
    if not all_seq_names:
        logging.info("No sequences available for prediction.")
    else:
        predictor.load_model(summary=False)

        strand_batches = _strand_batches(context.strands)
        max_strands_per_batch = max(len(strand_batch) for strand_batch in strand_batches)
        effective_group_target = max(1, GROUP_TARGET_SIZE // max_strands_per_batch)
        seq_groups = group_sequences(
            all_seq_names,
            all_seq_lens,
            target_size=effective_group_target,
            chunk_size=context.seq_len,
            parallel_factor=context.parallel_factor,
            upper_only=UPPER_ONLY,
            flank_size=context.flank_size,
        )
        total_groups = len(seq_groups) * len(strand_batches)
        logging.info(
            "%s planned %s sequence groups with adaptive chunk-size buckets.",
            log_prefix,
            len(seq_groups),
        )

        for group_index, seq_group in enumerate(seq_groups):
            genome_fasta.encode_sequences(seq=seq_group)
            for batch_index, strand_batch in enumerate(strand_batches):
                group_number = group_index * len(strand_batches) + batch_index + 1
                strand_label = "/".join(strand_batch)
                if strand_batch == ("+", "-"):
                    strand_label = "+/- batched"
                logging.info(
                    "%s prediction group %s/%s (%s)",
                    log_prefix,
                    group_number,
                    total_groups,
                    strand_label,
                )

                strand_inputs = []
                for strand in strand_batch:
                    x_data, coords, adapted_seqlen = predictor.load_genome_data(
                        genome_fasta,
                        seq_group,
                        upper_only=UPPER_ONLY,
                        strand=strand,
                        flank_size=context.flank_size,
                        encode=False,
                    )
                    strand_inputs.append(
                        {
                            "strand": strand,
                            "x_data": x_data,
                            "coords": coords,
                            "adapted_seqlen": adapted_seqlen,
                        }
                    )

                adapted_values = {item["adapted_seqlen"] for item in strand_inputs}
                if len(adapted_values) != 1:
                    raise ValueError(
                        "Positive and negative strand chunks produced different "
                        f"model lengths: {sorted(adapted_values)}."
                    )

                adapted_seqlen = strand_inputs[0]["adapted_seqlen"]
                predictor.adapt_batch_size(adapted_seqlen)
                if len(strand_inputs) == 1:
                    combined_chunks = strand_inputs[0]["x_data"]
                else:
                    chunk_counts = [item["x_data"].shape[0] for item in strand_inputs]
                    combined_chunks = np.concatenate(
                        [item["x_data"] for item in strand_inputs],
                        axis=0,
                    )
                    offset = 0
                    for item, chunk_count in zip(strand_inputs, chunk_counts):
                        item["x_data"] = combined_chunks[offset : offset + chunk_count]
                        offset += chunk_count
                del x_data, coords

                prediction_outputs = predictor.get_prediction_outputs(
                    combined_chunks,
                    hmm_filter=True,
                    clamsa_inp=None,
                    output_gene=context.output_gene,
                    output_repeat=context.output_repeat,
                )

                offset = 0
                for item in strand_inputs:
                    strand = item["strand"]
                    chunk_count = item["x_data"].shape[0]
                    output_slice = slice(offset, offset + chunk_count)
                    coords = item["coords"]
                    effective_chunks = _trim_prediction_flanks(
                        item["x_data"],
                        context.flank_size,
                    )

                    if context.output_gene:
                        effective_labels = _trim_prediction_flanks(
                            prediction_outputs["gene"][output_slice],
                            context.flank_size,
                        )
                        annotation, transcript_counter = predictor.create_gtf(
                            y_label=effective_labels,
                            coords=coords,
                            f_chunks=effective_chunks,
                            clamsa_inp=None,
                            strand=strand,
                            anno=annotation,
                            tx_id=transcript_counter,
                            filt=False,
                            allow_boundary_reprediction=context.flank_size == 0,
                        )
                        del effective_labels

                    if context.output_repeat:
                        effective_repeat = _trim_prediction_flanks(
                            prediction_outputs["repeat"][output_slice],
                            context.flank_size,
                        )
                        repeat_writer.add_predictions(
                            effective_repeat,
                            coords,
                            strand,
                            inference_seq_dict,
                        )
                        del effective_repeat

                    offset += chunk_count
                    del effective_chunks

                del combined_chunks, prediction_outputs, strand_inputs, adapted_seqlen

            genome_fasta.one_hot_encoded = None
            gc.collect()

    if coordinate_mapper is not None and coordinate_mapper.has_packing and context.output_gene:
        remap_stats = coordinate_mapper.remap_annotation(annotation)
        logging.info(
            "Remapped %s packed transcripts; marked %s transcripts for scaffold re-prediction.",
            remap_stats["remapped_transcripts"],
            remap_stats["dropped_packed_transcripts"],
        )
    else:
        remap_stats = {
            "remapped_transcripts": 0,
            "dropped_packed_transcripts": 0,
            "recheck_sequences": [],
        }

    return annotation, transcript_counter, remap_stats


def _subset_genome(genome, names):
    return {name: genome[name] for name in names if name in genome}


def _remove_annotation_sequences(annotation, seq_names):
    remove = set(seq_names)
    if not remove or annotation is None:
        return
    annotation.transcripts = {
        transcript_id: transcript
        for transcript_id, transcript in annotation.transcripts.items()
        if transcript.chr not in remove
    }
    annotation.genes = {"None": []}
    annotation.gene_gtf = {}


def run_prediction(args):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if getattr(args, "profile_hmm", False):
        os.environ["ORIONGENO_PROFILE_HMM"] = "1"

    start_time = time.time()
    checkpoint_path = _resolve_checkpoint_path(args)
    if not getattr(args, "genome", ""):
        logging.error("--genome or ORIONGENO_GENOME is required.")
        sys.exit(1)
    if not getattr(args, "out", ""):
        logging.error("--output or ORIONGENO_OUT is required.")
        sys.exit(1)
    genome_path = os.path.abspath(args.genome)
    output_path = os.path.abspath(args.out)
    output_dir = os.path.dirname(output_path)

    check_file_exists(checkpoint_path)
    check_file_exists(genome_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if should_skip_by_fasta_record_count(genome_path, DEFAULT_MAX_FASTA_RECORDS):
        return

    from .genome_anno import Anno

    seq_len = args.seq_len
    flank_size = args.flank_size
    check_flank_size(flank_size, seq_len)
    if not args.output_gene and not args.output_repeat:
        logging.error("At least one of --output-gene or --output-repeat must be True.")
        sys.exit(1)
    model_seq_len = seq_len + 2 * flank_size
    parallel_factor = _resolve_parallel_factor(args, model_seq_len, seq_len)
    check_parallel_factor(parallel_factor, model_seq_len, core_seq_len=seq_len)

    strands = PREDICTION_STRANDS
    species_name = getattr(args, "species_name", "")
    context = PredictionContext(
        checkpoint_path=checkpoint_path,
        seq_len=seq_len,
        flank_size=flank_size,
        model_seq_len=model_seq_len,
        batch_size=args.batch_size,
        strands=strands,
        species_name=species_name,
        parallel_factor=parallel_factor,
        output_gene=args.output_gene,
        output_repeat=args.output_repeat,
    )

    if species_name:
        logging.info("Species: %s", species_name)
    logging.info("Checkpoint: %s", checkpoint_path)
    logging.info("Genome FASTA: %s", genome_path)
    if args.output_gene:
        logging.info("Gene GTF output: %s", output_path)
    if args.output_repeat:
        logging.info("Repeat GTF output: %s", repeat_output_path(output_path))
    logging.info("Output window length: %s", seq_len)
    logging.info("Flank size per side: %s", flank_size)
    logging.info("Model input length: %s", model_seq_len)
    logging.info("Batch size: %s", args.batch_size)
    logging.info("Strands: %s", ",".join(strands))
    logging.info("Parallel factor: %s", parallel_factor)
    logging.info("Sequence group target size: %s", GROUP_TARGET_SIZE)
    logging.info("Use HMM: %s", USE_HMM)
    logging.info("Use uppercase sequence only: %s", UPPER_ONLY)

    genome = load_genome(genome_path)
    inference_genome, coordinate_mapper = prepare_inference_genome(
        genome,
        assembly_mode=ASSEMBLY_MODE,
        seq_len=seq_len,
        flank_size=flank_size,
        min_seq_len=MIN_SEQ_LEN,
        fragmented_record_threshold=FRAGMENTED_RECORD_THRESHOLD,
        pack_threshold=0,
        pack_spacer_len=PACK_SPACER_LEN,
        pack_target_size=PACK_TARGET_SIZE,
    )
    logging.info("Assembly inference mode: %s", coordinate_mapper.summary.get("mode", "native"))
    if coordinate_mapper.has_packing:
        logging.info(
            "Packed fragmented assembly: %s input records -> %s inference records.",
            coordinate_mapper.summary.get("kept_sequences"),
            coordinate_mapper.summary.get("inference_sequences"),
        )

    annotation = Anno(output_path, "oriongeno") if args.output_gene else None
    transcript_counter = 0
    output_seq_dict = {seq_name: len(seqrec.seq) for seq_name, seqrec in genome.items()}
    id_prefix = ""
    repeat_writer = (
        RepeatGtfWriter(
            output_path,
            id_prefix,
            coordinate_mapper=coordinate_mapper,
        )
        if args.output_repeat
        else None
    )

    annotation, transcript_counter, remap_stats = _predict_genome_records(
        inference_genome,
        context,
        annotation=annotation,
        transcript_counter=transcript_counter,
        repeat_writer=repeat_writer,
        coordinate_mapper=coordinate_mapper,
    )

    recheck_sequences = set(remap_stats["recheck_sequences"])
    if repeat_writer is not None:
        recheck_sequences.update(repeat_writer.consume_recheck_sequences())
    recheck_sequences = sorted(recheck_sequences)
    if recheck_sequences:
        logging.info(
            "Re-predicting %s original scaffolds whose packed predictions crossed N spacers.",
            len(recheck_sequences),
        )
        _remove_annotation_sequences(annotation, recheck_sequences)
        if repeat_writer is not None:
            repeat_writer.remove_sequences(recheck_sequences)
        recheck_genome = _subset_genome(genome, recheck_sequences)
        annotation, transcript_counter, _ = _predict_genome_records(
            recheck_genome,
            context,
            annotation=annotation,
            transcript_counter=transcript_counter,
            repeat_writer=repeat_writer,
            coordinate_mapper=None,
            log_prefix="OrionGeno recheck",
        )

    if args.output_gene:
        logging.info("Predicted transcripts before filtering: %s", len(annotation.transcripts))
        filter_and_write_outputs(
            annotation,
            genome,
            output_seq_dict,
            output_path,
            id_prefix,
        )

    if args.output_repeat:
        repeat_writer.write()

    duration = time.time() - start_time
    print(f"OrionGeno finished in {duration / 60:.4f} minutes.")


def build_predict_parser(prog="main.py"):
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run OrionGeno prediction from a genome FASTA and checkpoint.",
    )
    parser.add_argument(
        "--genome",
        default=env_str("ORIONGENO_GENOME", ""),
        help="Genome FASTA input path. Required unless ORIONGENO_GENOME is set.",
    )
    parser.add_argument(
        "--output",
        dest="out",
        default=env_str("ORIONGENO_OUT", ""),
        help="Gene GTF output path. Repeat output uses the same basename with .repeat.gtf.",
    )
    parser.add_argument(
        "--checkpoint",
        default=env_str("ORIONGENO_CHECKPOINT", ""),
        help="OrionGeno checkpoint directory. Required unless ORIONGENO_CHECKPOINT is set.",
    )
    parser.add_argument(
        "--length",
        dest="seq_len",
        type=int,
        default=env_int("ORIONGENO_SEQ_LEN", 512000),
        help="Output window length.",
    )
    parser.add_argument(
        "--flank",
        dest="flank_size",
        type=int,
        default=env_int("ORIONGENO_FLANK_SIZE", 0),
        help="Context bases added to each side of every output window.",
    )
    parser.add_argument("--batch-size", type=int, default=env_int("ORIONGENO_BATCH_SIZE", 8))
    parser.add_argument(
        "--hmm-parallel-factor",
        type=int,
        default=env_int("ORIONGENO_HMM_PARALLEL_FACTOR", 0),
        help="Override the HMM chunk-parallel factor. Use 0 to choose it automatically.",
    )
    parser.add_argument(
        "--profile-hmm",
        type=str_to_bool,
        default=env_bool("ORIONGENO_PROFILE_HMM", False),
        help="Print per-batch HMM timing breakdowns.",
    )
    parser.add_argument(
        "--output-gene",
        type=str_to_bool,
        default=env_bool("ORIONGENO_OUTPUT_GENE", True),
    )
    parser.add_argument(
        "--output-repeat",
        type=str_to_bool,
        default=env_bool("ORIONGENO_OUTPUT_REPEAT", False),
    )
    parser.add_argument(
        "--species-name",
        default=env_str("ORIONGENO_SPECIES_NAME", ""),
        help="Optional species name used only for species conditioning; it does not select a checkpoint.",
    )
    return parser
