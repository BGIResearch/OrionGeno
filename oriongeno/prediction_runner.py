"""OrionGeno prediction orchestration and parser."""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time

import numpy as np

from .cli_config import (
    AUTO_VALUE,
    auto_nonnegative_int,
    auto_positive_int,
    check_file_exists,
    env_auto_nonnegative_int,
    env_auto_positive_int,
    env_bool,
    env_int,
    env_str,
    is_auto,
    str_to_bool,
)
from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_FLANK_SIZE,
    DEFAULT_FRAGMENTED_RECORD_THRESHOLD,
    DEFAULT_GENE_FILTER_MODE,
    DEFAULT_HMM_DECODE_BATCH,
    DEFAULT_MAX_CHUNKS_PER_INFERENCE_GROUP,
    DEFAULT_MAX_FASTA_RECORDS,
    DEFAULT_PACK_SPACER_LEN,
    DEFAULT_PACK_TARGET_SIZE,
    DEFAULT_SEQUENCE_LENGTH,
    SEQUENCE_GROUP_SIZE,
)
from .gff_outputs import (
    RepeatGffWriter,
    annotation_output_format,
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


ASSEMBLY_MODE = "auto"
FRAGMENTED_RECORD_THRESHOLD = DEFAULT_FRAGMENTED_RECORD_THRESHOLD
PACK_SPACER_LEN = DEFAULT_PACK_SPACER_LEN
PACK_TARGET_SIZE = DEFAULT_PACK_TARGET_SIZE
MIN_SEQ_LEN = 0
PREDICTION_STRANDS = ("+", "-")
USE_HMM = True
UPPER_ONLY = True
GROUP_TARGET_SIZE = SEQUENCE_GROUP_SIZE
BATCH_BOTH_STRANDS = True
MAX_CHUNKS_PER_INFERENCE_GROUP = DEFAULT_MAX_CHUNKS_PER_INFERENCE_GROUP
AUTO_BATCH_MEMORY_FRACTION = 0.82
AUTO_BATCH_GIB_PER_BASE = 6.2e-6
AUTO_BATCH_BASE_GIB = 0.5
AUTO_BATCH_MAX = 64
AUTO_HMM_DECODE_BATCH = 16


def _load_runtime_env_config():
    """Load optional prediction tuning from env at runtime, not import time."""
    global ASSEMBLY_MODE
    global FRAGMENTED_RECORD_THRESHOLD
    global PACK_SPACER_LEN
    global PACK_TARGET_SIZE
    global GROUP_TARGET_SIZE
    global BATCH_BOTH_STRANDS
    global MAX_CHUNKS_PER_INFERENCE_GROUP

    ASSEMBLY_MODE = env_str("ORIONGENO_ASSEMBLY_MODE", "auto")
    FRAGMENTED_RECORD_THRESHOLD = env_int(
        "ORIONGENO_FRAGMENTED_RECORD_THRESHOLD",
        DEFAULT_FRAGMENTED_RECORD_THRESHOLD,
    )
    PACK_SPACER_LEN = env_int("ORIONGENO_PACK_SPACER_LEN", DEFAULT_PACK_SPACER_LEN)
    PACK_TARGET_SIZE = env_int("ORIONGENO_PACK_TARGET_SIZE", DEFAULT_PACK_TARGET_SIZE)
    GROUP_TARGET_SIZE = env_int("ORIONGENO_SEQUENCE_GROUP_SIZE", SEQUENCE_GROUP_SIZE)
    BATCH_BOTH_STRANDS = env_bool("ORIONGENO_BATCH_BOTH_STRANDS", True)
    MAX_CHUNKS_PER_INFERENCE_GROUP = env_int(
        "ORIONGENO_MAX_CHUNKS_PER_INFERENCE_GROUP",
        DEFAULT_MAX_CHUNKS_PER_INFERENCE_GROUP,
    )


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _resolve_auto_batch_size(requested_batch_size, model_seq_len):
    """Resolve an explicit or automatic model-forward batch size."""
    if not is_auto(requested_batch_size):
        return int(requested_batch_size)

    try:
        import torch
    except ImportError:
        logging.warning("PyTorch is unavailable while resolving batch_size=auto; using batch_size=1.")
        return 1

    if not torch.cuda.is_available():
        logging.info("Resolved batch_size=auto to 1 because CUDA is unavailable.")
        return 1

    fraction = _env_float("ORIONGENO_AUTO_BATCH_MEMORY_FRACTION", AUTO_BATCH_MEMORY_FRACTION)
    fraction = min(max(fraction, 0.1), 0.95)
    gib_per_base = _env_float("ORIONGENO_AUTO_BATCH_GIB_PER_BASE", AUTO_BATCH_GIB_PER_BASE)
    gib_per_base = max(gib_per_base, 1e-7)
    base_gib = _env_float("ORIONGENO_AUTO_BATCH_BASE_GIB", AUTO_BATCH_BASE_GIB)
    max_batch = max(1, env_int("ORIONGENO_AUTO_BATCH_MAX", AUTO_BATCH_MAX))

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_gib = free_bytes / 1024**3
        total_gib = total_bytes / 1024**3
    except RuntimeError:
        device = torch.cuda.current_device()
        total_gib = torch.cuda.get_device_properties(device).total_memory / 1024**3
        free_gib = total_gib

    per_chunk_gib = max(model_seq_len * gib_per_base, 0.25)
    usable_gib = max(0.0, free_gib * fraction - base_gib)
    batch_size = int(usable_gib // per_chunk_gib)
    batch_size = max(1, min(batch_size, max_batch))
    logging.info(
        "Resolved batch_size=auto to %s (free %.2f/%.2f GiB, model input length %s).",
        batch_size,
        free_gib,
        total_gib,
        model_seq_len,
    )
    return batch_size


def _resolve_auto_hmm_decode_batch(requested_decode_batch, batch_size):
    """Resolve HMM decode batch; 0 keeps legacy reuse of the model batch."""
    if not is_auto(requested_decode_batch):
        return int(requested_decode_batch)
    decode_batch = max(batch_size, env_int("ORIONGENO_AUTO_HMM_DECODE_BATCH", AUTO_HMM_DECODE_BATCH))
    logging.info("Resolved hmm_decode_batch=auto to %s.", decode_batch)
    return decode_batch


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
        gene_filter_mode,
        hmm_decode_batch=0,
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
        self.gene_filter_mode = gene_filter_mode
        self.hmm_decode_batch = hmm_decode_batch


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
    if BATCH_BOTH_STRANDS and strands == ("+", "-"):
        return [strands]
    return [(strand,) for strand in strands]


def _as_numpy(array):
    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _internal_chunk_span(strand_count):
    if MAX_CHUNKS_PER_INFERENCE_GROUP <= 0:
        return None
    return max(1, MAX_CHUNKS_PER_INFERENCE_GROUP // max(1, strand_count))


def _predict_internal_chunks(
    predictor,
    strand_inputs,
    context,
    *,
    log_prefix,
    group_number,
    total_groups,
    strand_label,
):
    max_chunk_count = max(item["x_data"].shape[0] for item in strand_inputs)
    chunk_span = _internal_chunk_span(len(strand_inputs))
    if chunk_span is None or chunk_span >= max_chunk_count:
        chunk_span = max_chunk_count
    internal_total = max(1, int(np.ceil(max_chunk_count / chunk_span)))
    if internal_total > 1:
        logging.info(
            "%s prediction group %s/%s (%s) split into %s internal chunk windows "
            "of up to %s chunks per strand.",
            log_prefix,
            group_number,
            total_groups,
            strand_label,
            internal_total,
            chunk_span,
        )

    for internal_index, start in enumerate(range(0, max_chunk_count, chunk_span), start=1):
        end = min(start + chunk_span, max_chunk_count)
        sub_items = []
        sub_chunks = []
        for item in strand_inputs:
            item_chunk_count = item["x_data"].shape[0]
            if start >= item_chunk_count:
                continue
            item_end = min(end, item_chunk_count)
            sub_items.append((item, start, item_end))
            sub_chunks.append(item["x_data"][start:item_end])
        if not sub_chunks:
            continue

        if len(sub_chunks) == 1:
            combined_chunks = sub_chunks[0]
        else:
            combined_chunks = np.concatenate(sub_chunks, axis=0)
        if internal_total > 1:
            logging.info(
                "%s prediction group %s/%s (%s) internal chunk %s/%s shape=%s",
                log_prefix,
                group_number,
                total_groups,
                strand_label,
                internal_index,
                internal_total,
                combined_chunks.shape,
            )

        prediction_outputs = predictor.get_prediction_outputs(
            combined_chunks,
            output_gene=context.output_gene,
            output_repeat=context.output_repeat,
        )

        offset = 0
        for item, item_start, item_end in sub_items:
            chunk_count = item_end - item_start
            item_outputs = item.setdefault("prediction_outputs", {})
            output_slice = slice(offset, offset + chunk_count)
            if context.output_gene:
                gene_chunk = _as_numpy(prediction_outputs["gene"][output_slice])
                gene_output = item_outputs.get("gene")
                if gene_output is None:
                    gene_output = np.empty(
                        (item["x_data"].shape[0],) + gene_chunk.shape[1:],
                        dtype=gene_chunk.dtype,
                    )
                    item_outputs["gene"] = gene_output
                gene_output[item_start:item_end] = gene_chunk
            if context.output_repeat:
                repeat_chunk = _as_numpy(prediction_outputs["repeat"][output_slice])
                repeat_output = item_outputs.get("repeat")
                if repeat_output is None:
                    repeat_output = np.empty(
                        (item["x_data"].shape[0],) + repeat_chunk.shape[1:],
                        dtype=repeat_chunk.dtype,
                    )
                    item_outputs["repeat"] = repeat_output
                repeat_output[item_start:item_end] = repeat_chunk
            offset += chunk_count

        del combined_chunks, prediction_outputs, sub_chunks, sub_items
        gc.collect()


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
    from .prediction_annotator import PredictionAnnotator

    inference_seq_dict = {}

    predictor = PredictionAnnotator(
        model_path=context.checkpoint_path,
        seq_len=context.seq_len,
        batch_size=context.batch_size,
        hmm=USE_HMM and context.output_gene,
        temp_dir=None,
        genome=genome,
        upper_only=UPPER_ONLY,
        species_name=context.species_name,
        strand="+",
        parallel_factor=context.parallel_factor,
        hmm_decode_batch=context.hmm_decode_batch,
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
                del x_data, coords

                _predict_internal_chunks(
                    predictor,
                    strand_inputs,
                    context,
                    log_prefix=log_prefix,
                    group_number=group_number,
                    total_groups=total_groups,
                    strand_label=strand_label,
                )

                for item in strand_inputs:
                    strand = item["strand"]
                    coords = item["coords"]
                    effective_chunks = _trim_prediction_flanks(
                        item["x_data"],
                        context.flank_size,
                    )

                    if context.output_gene:
                        effective_labels = _trim_prediction_flanks(
                            item["prediction_outputs"]["gene"],
                            context.flank_size,
                        )
                        annotation, transcript_counter = predictor.create_gene_annotation(
                            y_label=effective_labels,
                            coords=coords,
                            f_chunks=effective_chunks,
                            strand=strand,
                            anno=annotation,
                            tx_id=transcript_counter,
                            filt=False,
                            allow_boundary_reprediction=context.flank_size == 0,
                        )
                        del effective_labels

                    if context.output_repeat:
                        effective_repeat = _trim_prediction_flanks(
                            item["prediction_outputs"]["repeat"],
                            context.flank_size,
                        )
                        repeat_writer.add_predictions(
                            effective_repeat,
                            coords,
                            strand,
                            inference_seq_dict,
                        )
                        del effective_repeat

                    del effective_chunks

                del strand_inputs, adapted_seqlen

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
    _load_runtime_env_config()
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

    from .genome_annotation import Anno

    seq_len = args.seq_len
    flank_size = args.flank_size
    check_flank_size(flank_size, seq_len)
    if not args.output_gene and not args.output_repeat:
        logging.error("At least one of --output-gene or --output-repeat must be True.")
        sys.exit(1)
    model_seq_len = seq_len + 2 * flank_size
    parallel_factor = _resolve_parallel_factor(args, model_seq_len, seq_len)
    check_parallel_factor(parallel_factor, model_seq_len, core_seq_len=seq_len)
    batch_size = _resolve_auto_batch_size(args.batch_size, model_seq_len)
    hmm_decode_batch = _resolve_auto_hmm_decode_batch(
        getattr(args, "hmm_decode_batch", DEFAULT_HMM_DECODE_BATCH),
        batch_size,
    )

    strands = PREDICTION_STRANDS
    species_name = getattr(args, "species_name", "")
    context = PredictionContext(
        checkpoint_path=checkpoint_path,
        seq_len=seq_len,
        flank_size=flank_size,
        model_seq_len=model_seq_len,
        batch_size=batch_size,
        strands=strands,
        species_name=species_name,
        parallel_factor=parallel_factor,
        output_gene=args.output_gene,
        output_repeat=args.output_repeat,
        gene_filter_mode=args.gene_filter_mode,
        hmm_decode_batch=hmm_decode_batch,
    )

    if species_name:
        logging.info("Species: %s", species_name)
    logging.info("Checkpoint: %s", checkpoint_path)
    logging.info("Genome FASTA: %s", genome_path)
    if args.output_gene:
        logging.info("Gene %s output: %s", annotation_output_format(output_path), output_path)
    if args.output_repeat:
        repeat_path = repeat_output_path(output_path)
        logging.info("Repeat %s output: %s", annotation_output_format(repeat_path), repeat_path)
    logging.info("Output window length: %s", seq_len)
    logging.info("Flank size per side: %s", flank_size)
    logging.info("Model input length: %s", model_seq_len)
    logging.info("Batch size: %s", batch_size)
    logging.info("Strands: %s", ",".join(strands))
    logging.info("Parallel factor: %s", parallel_factor)
    logging.info(
        "HMM decode batch: %s",
        context.hmm_decode_batch or f"{batch_size} (model-forward batch)",
    )
    logging.info("Gene filter mode: %s", args.gene_filter_mode)
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
        RepeatGffWriter(
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
            gene_filter_mode=args.gene_filter_mode,
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
        help="Gene GFF output path. Repeat output uses the same basename with .repeat.gff.",
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
        default=env_int("ORIONGENO_SEQ_LEN", DEFAULT_SEQUENCE_LENGTH),
        help="Output window length.",
    )
    parser.add_argument(
        "--flank",
        dest="flank_size",
        type=int,
        default=env_int("ORIONGENO_FLANK_SIZE", DEFAULT_FLANK_SIZE),
        help="Context bases added to each side of every output window.",
    )
    parser.add_argument(
        "--batch-size",
        type=auto_positive_int,
        default=env_auto_positive_int("ORIONGENO_BATCH_SIZE", DEFAULT_BATCH_SIZE),
        help="Model-forward batch size. Use 'auto' to estimate from available GPU memory.",
    )
    parser.add_argument(
        "--hmm-parallel-factor",
        type=int,
        default=env_int("ORIONGENO_HMM_PARALLEL_FACTOR", 0),
        help="Override the HMM chunk-parallel factor. Use 0 to choose it automatically.",
    )
    parser.add_argument(
        "--hmm-decode-batch",
        type=auto_nonnegative_int,
        default=env_auto_nonnegative_int(
            "ORIONGENO_HMM_DECODE_BATCH",
            DEFAULT_HMM_DECODE_BATCH,
        ),
        help=(
            "HMM Viterbi decode batch size, decoupled from --batch-size. Larger "
            "values improve CPU Viterbi batch-level parallelism and amortize "
            "per-step CUDA kernel-launch overhead on the GPU fallback path; "
            "memory grows with the batch and a CUDA OOM is handled by halving "
            "and retrying. Use 'auto' for a conservative tuned value; use 0 to "
            "reuse the model-forward batch size (legacy behavior)."
        ),
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
        "--gene-filter-mode",
        choices=("strict", "none"),
        default=env_str(
            "ORIONGENO_GENE_FILTER_MODE",
            DEFAULT_GENE_FILTER_MODE,
        ).lower(),
        help=(
            "Gene annotation filtering mode. 'strict' writes only the strict-filtered "
            "records to --output; 'none' writes unfiltered predictions to --output."
        ),
    )
    parser.add_argument(
        "--species-name",
        default=env_str("ORIONGENO_SPECIES_NAME", ""),
        help="Optional species name used only for species conditioning; it does not select a checkpoint.",
    )
    return parser
