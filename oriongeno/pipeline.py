"""Inference pipeline orchestration for OrionGeno.

Authors: wangshengfu, caixudong
"""

import logging
import os
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist

from .eval_model_class import PredictionGTF
from .genome_anno import Anno
from .genome_utils import (
    SEQGROUP_SIZE,
    build_genome_seq_dict,
    check_file_exists,
    check_parallel_factor,
    check_seq_len,
    group_sequences,
    load_genome,
    prepare_inference_genome,
    remap_annotation_from_packed_sequences,
    remap_repeat_intervals_from_packed_sequences,
)
from .output_utils import (
    ensure_parent_dir,
    namespace_transcripts,
    resolve_model_bundle_paths,
    resolve_output_paths,
    write_filtered_gene_outputs,
    write_repeat_outputs,
)
from .runtime_utils import (
    RuntimeContext,
    cleanup_runtime,
    configure_rank_logging,
    dist_barrier,
    emit_rank_progress,
    format_sequence_group_label,
    load_all_rank_payloads,
    prepare_dist_work_dir,
    save_rank_payload,
    shard_work,
)
from .species_router import (
    DEFAULT_MODEL_ROOT,
    DEFAULT_SPECIES_EMBEDDING_PATH,
    DEFAULT_SPECIES_TABLE_PATH,
    normalize_species_name,
    resolve_species_runtime_assets,
)


def setup_runtime(args, output_path):
    """Initialize device placement and optional torch.distributed state."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = args.local_rank
    if local_rank < 0:
        local_rank = int(os.environ.get("LOCAL_RANK", rank))

    requested_device = args.device.lower()
    if requested_device == "auto":
        if torch.cuda.is_available():
            if local_rank < 0:
                local_rank = 0
            device = f"cuda:{local_rank}"
        else:
            device = "cpu"
    elif requested_device == "cuda":
        if not torch.cuda.is_available():
            logging.error(
                'ERROR: "--device cuda" was requested, but CUDA is not available.'
            )
            sys.exit(1)
        if local_rank < 0:
            local_rank = 0
        device = f"cuda:{local_rank}"
    elif requested_device.startswith("cuda:") and world_size > 1:
        if not torch.cuda.is_available():
            logging.error(
                "ERROR: A CUDA device was requested, but CUDA is not available."
            )
            sys.exit(1)
        if local_rank < 0:
            local_rank = 0
        device = f"cuda:{local_rank}"
    else:
        device = requested_device

    backend = ""
    if world_size > 1:
        timeout = timedelta(minutes=args.dist_timeout_minutes)
        backend = args.dist_backend or ("nccl" if device.startswith("cuda") else "gloo")
        if device.startswith("cuda"):
            torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, timeout=timeout)
    elif device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(int(device.split(":")[-1]))

    dist_tmp_dir = (
        os.path.abspath(args.dist_tmp_dir)
        if args.dist_tmp_dir
        else os.path.join(
            os.path.dirname(output_path),
            f".{os.path.basename(output_path)}.dist",
        )
    )
    return RuntimeContext(
        enabled=world_size > 1,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        backend=backend,
        work_dir=dist_tmp_dir,
    )


def run_annotation_workflow(
    args,
    runtime,
    genome,
    original_genome,
    original_genome_seq_dict,
    packing_info,
    gtf_out,
    repeat_out,
    gene_model_path,
    repeat_checkpoint_path,
    gene_checkpoint_format,
    repeat_checkpoint_format,
    gene_config_path,
    repeat_config_path,
    repeat_classifier_path,
    spe_embedding_path,
    seq_len,
    flank_bp,
    batch_size,
    min_seq_len,
    use_hmm,
    use_spe_embeddings,
    species_name,
    parallel_factor,
    strand,
    repeat_min_run_length,
    repeat_max_gap,
):
    """Run gene/repeat inference on the local shard and emit merged outputs."""
    output_gene = args.use_gene_annotation
    output_repeat = args.use_repeat_annotation
    requested_gene_strands = list(strand)
    workflow_strands = list(requested_gene_strands)
    if output_repeat and "+" not in workflow_strands:
        workflow_strands.append("+")

    anno = Anno(gtf_out, "anno") if output_gene else None
    tx_id = 0
    repeat_intervals = []
    inference_genome_seq_dict = build_genome_seq_dict(genome)

    for j, strand_name in enumerate(workflow_strands):
        run_gene_for_strand = output_gene and strand_name in requested_gene_strands
        run_repeat_for_strand = output_repeat and strand_name == "+"

        if not run_gene_for_strand and not run_repeat_for_strand:
            continue

        pred_gene_gtf = None
        pred_repeat_gtf = None

        if run_gene_for_strand:
            pred_gene_gtf = PredictionGTF(
                model_path=gene_model_path,
                repeat_model_path=repeat_checkpoint_path,
                gene_checkpoint_format=gene_checkpoint_format,
                repeat_checkpoint_format=repeat_checkpoint_format,
                gene_config_path=gene_config_path,
                repeat_config_path=repeat_config_path,
                repeat_classifier_path=repeat_classifier_path,
                spe_embedding_path=spe_embedding_path,
                seq_len=seq_len,
                flank_bp=flank_bp,
                batch_size=batch_size,
                hmm=use_hmm and output_gene,
                genome=genome,
                use_onlyUpper=True,
                use_spe_embeddings=use_spe_embeddings,
                species_name=species_name,
                parallel_factor=parallel_factor,
                device=runtime.device,
                verbose=False,
            )

        if run_repeat_for_strand:
            pred_repeat_gtf = PredictionGTF(
                model_path=gene_model_path,
                repeat_model_path=repeat_checkpoint_path,
                gene_checkpoint_format=gene_checkpoint_format,
                repeat_checkpoint_format=repeat_checkpoint_format,
                gene_config_path=gene_config_path,
                repeat_config_path=repeat_config_path,
                repeat_classifier_path=repeat_classifier_path,
                spe_embedding_path=spe_embedding_path,
                seq_len=seq_len,
                flank_bp=flank_bp,
                batch_size=batch_size,
                hmm=False,
                genome=genome,
                use_onlyUpper=True,
                use_spe_embeddings=use_spe_embeddings,
                species_name=species_name,
                parallel_factor=parallel_factor,
                device=runtime.device,
                verbose=False,
            )

        group_predictor = pred_gene_gtf or pred_repeat_gtf
        genome_fasta = group_predictor.init_fasta(
            chunk_len=seq_len,
            min_seq_len=min_seq_len,
        )
        seq_groups = group_sequences(
            genome_fasta.sequence_names,
            [len(sequence) for sequence in genome_fasta.sequences],
            t=SEQGROUP_SIZE,
            chunk_size=seq_len + 2 * flank_bp,
        )
        local_seq_groups = shard_work(seq_groups, runtime.rank, runtime.world_size)
        if not local_seq_groups:
            continue

        if run_repeat_for_strand:
            pred_repeat_gtf.load_repeat_annotation_model(
                summary=runtime.is_main and j == 0 and not run_gene_for_strand
            )
            for local_idx, seq_group in enumerate(local_seq_groups):
                emit_rank_progress(
                    runtime,
                    "OrionGeno repeat annotation "
                    + f"{local_idx + 1}/{len(local_seq_groups)} in strand-independent mode "
                    + f"(internal '{strand_name}' pass): "
                    + f"{format_sequence_group_label(seq_group)}",
                )
                repeat_genome_fasta = pred_repeat_gtf.init_fasta(
                    chunk_len=seq_len,
                    min_seq_len=min_seq_len,
                )
                x_data, coords, adapted_seqlen = pred_repeat_gtf.load_genome_data(
                    repeat_genome_fasta,
                    seq_group,
                    strand=strand_name,
                )
                pred_repeat_gtf.adapt_batch_size(adapted_seqlen)
                repeat_pred = pred_repeat_gtf.get_repeat_annotations(x_data)
                repeat_pred = pred_repeat_gtf.crop_to_core_region(repeat_pred)
                repeat_intervals.extend(
                    pred_repeat_gtf.create_repeat_intervals(
                        repeat_pred,
                        coords,
                        sequence_lengths=inference_genome_seq_dict,
                    )
                )
            pred_repeat_gtf.release_runtime_model()

        if run_gene_for_strand:
            pred_gene_gtf.load_gene_annotation_model(
                summary=runtime.is_main and j == 0
            )
            for local_idx, seq_group in enumerate(local_seq_groups):
                emit_rank_progress(
                    runtime,
                    "OrionGeno gene annotation "
                    + f"{local_idx + 1}/{len(local_seq_groups)} on strand {strand_name}: "
                    + f"{format_sequence_group_label(seq_group)}",
                )
                gene_genome_fasta = pred_gene_gtf.init_fasta(
                    chunk_len=seq_len,
                    min_seq_len=min_seq_len,
                )
                x_data, coords, adapted_seqlen = pred_gene_gtf.load_genome_data(
                    gene_genome_fasta,
                    seq_group,
                    strand=strand_name,
                )
                pred_gene_gtf.adapt_batch_size(adapted_seqlen)
                hmm_pred = pred_gene_gtf.get_gene_annotations(x_data)
                hmm_pred = pred_gene_gtf.crop_to_core_region(hmm_pred)
                anno, tx_id = pred_gene_gtf.create_gtf(
                    y_label=hmm_pred,
                    coords=coords,
                    strand=strand_name,
                    anno=anno,
                    tx_id=tx_id,
                    sequence_lengths=inference_genome_seq_dict,
                )
            pred_gene_gtf.release_runtime_model()

    if output_gene:
        anno = remap_annotation_from_packed_sequences(anno, packing_info)
    if output_repeat:
        repeat_intervals = remap_repeat_intervals_from_packed_sequences(
            repeat_intervals,
            packing_info,
        )

    if runtime.enabled:
        save_rank_payload(
            runtime,
            "genes",
            {
                "transcripts": anno.transcripts if output_gene else {},
                "genome_seq_dict": original_genome_seq_dict if output_gene else {},
                "repeat_intervals": repeat_intervals if output_repeat else [],
            },
        )
        dist_barrier(runtime)
        if runtime.is_main:
            merged_anno = Anno("", "anno") if output_gene else None
            merged_genome_seq_dict = {}
            merged_repeat_intervals = []
            for rank, payload in load_all_rank_payloads(runtime, "genes"):
                if output_gene:
                    merged_genome_seq_dict.update(payload.get("genome_seq_dict", {}))
                    merged_anno.add_transcripts(
                        namespace_transcripts(
                            payload.get("transcripts", {}),
                            f"r{rank}",
                        )
                    )
                if output_repeat:
                    merged_repeat_intervals.extend(payload.get("repeat_intervals", []))
            if output_gene:
                write_filtered_gene_outputs(
                    merged_anno,
                    original_genome,
                    merged_genome_seq_dict,
                    gtf_out,
                    args.id_prefix,
                    include_utr=args.include_utr,
                )
            if output_repeat:
                write_repeat_outputs(
                    merged_repeat_intervals,
                    repeat_out,
                    min_repeat_bases=repeat_min_run_length,
                    max_nonrepeat_gap_bases=repeat_max_gap,
                )
        dist_barrier(runtime)
        return

    if output_gene:
        write_filtered_gene_outputs(
            anno,
            original_genome,
            original_genome_seq_dict,
            gtf_out,
            args.id_prefix,
            include_utr=args.include_utr,
        )
    if output_repeat:
        write_repeat_outputs(
            repeat_intervals,
            repeat_out,
            min_repeat_bases=repeat_min_run_length,
            max_nonrepeat_gap_bases=repeat_max_gap,
        )


def log_run_configuration(
    args,
    runtime,
    output_gene,
    output_repeat,
    gtf_out,
    repeat_out,
    batch_size,
    seq_len,
    flank_bp,
    model_input_seq_len,
    min_seq_len,
    strand,
    parallel_factor,
    use_spe_embeddings,
    species_name,
    genome_path,
):
    """Print one consolidated summary of the active runtime configuration."""
    if not runtime.is_main:
        return

    logging.info("Model interface: OrionGeno unified species-conditioned runtime")
    logging.info(f"Gene annotation enabled: {output_gene}")
    logging.info(f"Repeat annotation enabled: {output_repeat}")
    if output_gene:
        logging.info(f"Gene annotation GTF: {gtf_out}")
    if output_repeat:
        logging.info(f"Repeat annotation GTF: {repeat_out}")
        if args.repeat_min_run_length > 1:
            logging.info(
                f"Repeat minimum positive run length: {args.repeat_min_run_length}"
            )
        if args.repeat_max_gap > 0:
            logging.info(
                f"Repeat short-gap smoothing: fill non-repeat gaps <= {args.repeat_max_gap} bp."
            )
        logging.info(
            "Repeat annotation is strand-independent. OrionGeno runs one internal "
            "'+' pass for repeat output only."
        )
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Core window length: {seq_len}")
    logging.info(f"Flanking context on each side: {flank_bp} bp")
    logging.info(f"Total model input length: {model_input_seq_len}")
    if flank_bp > 0:
        logging.info(
            "Flank context mode enabled: each window includes left and right context, "
            "but only the center region is written to the final output."
        )
    else:
        logging.info(
            "Flank context mode disabled: each window is annotated directly."
        )
    logging.info(f"Minimum sequence length: {min_seq_len}")
    logging.info(f"Strand: {strand}")
    if use_spe_embeddings and species_name:
        logging.info("Species embeddings enabled: True")
        logging.info(f"Species embedding name: {species_name}")
    elif use_spe_embeddings and not species_name:
        logging.info("Species embeddings enabled: True")
        logging.info(
            "Species embeddings were requested without species_name; continuing "
            "without species embeddings."
        )
    logging.info(f"Genome sequence path: {genome_path}")
    if args.sequence_name_include_regex:
        logging.info(f"Sequence level filter: {args.sequence_level}")
        logging.info(f"Sequence include regex: {args.sequence_name_include_regex}")
    if args.sequence_name_exclude_regex:
        if not args.sequence_name_include_regex:
            logging.info(f"Sequence level filter: {args.sequence_level}")
        logging.info(f"Sequence exclude regex: {args.sequence_name_exclude_regex}")
    if args.scaffold_pack_mode != "auto":
        logging.info(f"Scaffold pack mode: {args.scaffold_pack_mode}")
    if args.scaffold_pack_gap_bp != 4096:
        logging.info(f"Scaffold pack gap bp: {args.scaffold_pack_gap_bp}")
    if args.scaffold_pack_target_bp != SEQGROUP_SIZE:
        logging.info(f"Scaffold pack target bp: {args.scaffold_pack_target_bp}")
    logging.info(f"Include UTR features in output: {args.include_utr}")
    logging.info(f"Runtime device: {runtime.device}")
    if runtime.enabled:
        logging.info(
            "Distributed inference enabled: "
            f"world_size={runtime.world_size}, backend={runtime.backend}, "
            f"shared_dir={runtime.work_dir}"
        )
        logging.info(
            f"Distributed timeout (minutes): {args.dist_timeout_minutes}"
        )


def run_oriongeno(args):
    """Execute the OrionGeno command-line pipeline."""
    start_time = time.time()
    requested_out = os.path.abspath(args.out)
    output_gene = args.use_gene_annotation
    output_repeat = args.use_repeat_annotation
    species_name = normalize_species_name(getattr(args, "species_name", ""))
    if not hasattr(args, "repeat_min_run_length"):
        args.repeat_min_run_length = 1
    if not hasattr(args, "repeat_max_gap"):
        args.repeat_max_gap = 30

    if not output_gene and not output_repeat:
        logging.error(
            'ERROR: At least one of "--use_gene_annotation" or "--use_repeat_annotation" must be True.'
        )
        sys.exit(1)
    if args.repeat_min_run_length < 1:
        logging.error('ERROR: "--repeat_min_run_length" must be at least 1.')
        sys.exit(1)
    if args.repeat_max_gap < 0:
        logging.error('ERROR: "--repeat_max_gap" must be >= 0.')
        sys.exit(1)
    if not species_name:
        logging.error('ERROR: The argument "--species_name" is required.')
        sys.exit(1)

    model_override_path = getattr(args, "model", "").strip()
    model_root = (
        os.path.abspath(args.model_root)
        if getattr(args, "model_root", "")
        else DEFAULT_MODEL_ROOT
    )
    species_table_path = (
        os.path.abspath(args.species_table_path)
        if getattr(args, "species_table_path", "")
        else DEFAULT_SPECIES_TABLE_PATH
    )
    spe_embedding_path = (
        os.path.abspath(args.species_embedding_path)
        if getattr(args, "species_embedding_path", "")
        else DEFAULT_SPECIES_EMBEDDING_PATH
    )

    try:
        if model_override_path:
            model_bundle_paths = resolve_model_bundle_paths(model_override_path)
            model_bundle_paths["species_embedding_path"] = spe_embedding_path
        else:
            model_bundle_paths = resolve_species_runtime_assets(
                species_name=species_name,
                model_root=model_root,
                species_table_path=species_table_path,
                species_embedding_path=spe_embedding_path,
            )
            species_name = model_bundle_paths["species_name"]
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        logging.error(f"ERROR: {exc}")
        sys.exit(1)

    gene_model_path = model_bundle_paths["gene_model_path"]
    repeat_checkpoint_path = model_bundle_paths["repeat_model_path"]
    gene_checkpoint_format = model_bundle_paths["gene_checkpoint_format"]
    repeat_checkpoint_format = model_bundle_paths["repeat_checkpoint_format"]
    gene_config_path = model_bundle_paths["gene_config_path"]
    repeat_config_path = model_bundle_paths["repeat_config_path"]
    repeat_classifier_path = model_bundle_paths["repeat_classifier_path"]
    spe_embedding_path = model_bundle_paths["species_embedding_path"]
    if output_gene and not gene_model_path:
        logging.error(
            "ERROR: Gene annotation was requested, but the internal OrionGeno "
            "runtime assets do not contain the files needed for gene annotation."
        )
        sys.exit(1)
    if output_repeat and not repeat_checkpoint_path:
        logging.error(
            "ERROR: Repeat annotation was requested, but the internal OrionGeno "
            "runtime assets do not contain the files needed for repeat annotation."
        )
        sys.exit(1)

    gtf_out, repeat_out = resolve_output_paths(
        requested_out,
        args.repeat_out,
        output_gene,
        output_repeat,
    )
    if output_gene and output_repeat:
        gene_out_abs = os.path.abspath(gtf_out)
        repeat_out_abs = os.path.abspath(repeat_out)
        if gene_out_abs == repeat_out_abs:
            logging.error(
                "ERROR: Gene and repeat outputs resolve to the same file path. "
                "Please provide different output targets."
            )
            sys.exit(1)
    if output_gene:
        ensure_parent_dir(gtf_out)
    if output_repeat:
        ensure_parent_dir(repeat_out)

    runtime_output_path = gtf_out or repeat_out or requested_out
    runtime = setup_runtime(args, runtime_output_path)
    configure_rank_logging(runtime)
    if runtime.enabled:
        prepare_dist_work_dir(runtime)

    try:
        use_hmm = args.use_hmm
        use_spe_embeddings = True

        batch_size = args.batch_size
        seq_len = args.seq_len
        flank_bp = args.flank_bp
        min_seq_len = args.min_genome_seqlen
        if batch_size < 1:
            logging.error('ERROR: The argument "batch_size" has to be > 0.')
            sys.exit(1)
        check_seq_len(seq_len)
        if flank_bp < 0:
            logging.error('ERROR: The argument "flank_bp" has to be >= 0.')
            sys.exit(1)
        model_input_seq_len = seq_len + 2 * flank_bp
        if args.scaffold_pack_gap_bp < 0:
            logging.error('ERROR: The argument "scaffold_pack_gap_bp" has to be >= 0.')
            sys.exit(1)
        if args.scaffold_pack_target_bp < 1:
            logging.error(
                'ERROR: The argument "scaffold_pack_target_bp" has to be > 0.'
            )
            sys.exit(1)
        if args.dist_timeout_minutes < 1:
            logging.error(
                'ERROR: The argument "dist_timeout_minutes" has to be > 0.'
            )
            sys.exit(1)

        strand = [value for value in args.strand.split(",") if value in ["+", "-"]]
        if not strand:
            logging.error(
                'ERROR: The argument "strand" has to be either "+" or "-" or "+,-". '
                f"Current value: {args.strand}."
            )
            sys.exit(1)
        if output_repeat and not output_gene:
            strand = ["+"]
            if runtime.is_main:
                logging.info(
                    "Repeat-only mode uses strand '+' only because repeat output is strand-independent."
                )
        elif output_repeat and "+" not in strand and runtime.is_main:
            logging.info(
                "Repeat annotation is strand-independent, so '+' will be used internally once for repeat output."
            )

        parallel_factor = check_parallel_factor(
            args.parallel_factor,
            model_input_seq_len,
        )

        genome_path = os.path.abspath(args.genome)
        check_file_exists(genome_path)

        log_run_configuration(
            args=args,
            runtime=runtime,
            output_gene=output_gene,
            output_repeat=output_repeat,
            gtf_out=gtf_out,
            repeat_out=repeat_out,
            batch_size=batch_size,
            seq_len=seq_len,
            flank_bp=flank_bp,
            model_input_seq_len=model_input_seq_len,
            min_seq_len=min_seq_len,
            strand=strand,
            parallel_factor=parallel_factor,
            use_spe_embeddings=use_spe_embeddings,
            species_name=species_name,
            genome_path=genome_path,
        )

        genome, total_records, kept_records = load_genome(
            genome_path,
            sequence_level=args.sequence_level,
            include_regex=args.sequence_name_include_regex,
            exclude_regex=args.sequence_name_exclude_regex,
        )
        if kept_records == 0:
            logging.error(
                "ERROR: No FASTA records matched the current sequence filtering settings."
            )
            sys.exit(1)
        if runtime.is_main:
            logging.info(
                f"Loaded {kept_records}/{total_records} FASTA records after filtering."
            )

        original_genome = genome
        original_genome_seq_dict = build_genome_seq_dict(original_genome)
        inference_genome, packing_info = prepare_inference_genome(
            original_genome,
            scaffold_pack_mode=args.scaffold_pack_mode,
            scaffold_pack_gap_bp=args.scaffold_pack_gap_bp,
            scaffold_pack_target_bp=args.scaffold_pack_target_bp,
        )
        if runtime.is_main:
            if packing_info.get("enabled"):
                logging.info(
                    "Scaffold packing enabled: "
                    f"{packing_info['original_scaffold_count']} scaffold-like sequences "
                    f"packed into {packing_info['packed_count']} pseudo-sequences "
                    f"(gap={packing_info['spacer_bp']} bp, "
                    f"target={packing_info['target_bp']} bp)."
                )

        run_annotation_workflow(
            args=args,
            runtime=runtime,
            genome=inference_genome,
            original_genome=original_genome,
            original_genome_seq_dict=original_genome_seq_dict,
            packing_info=packing_info,
            gtf_out=gtf_out,
            repeat_out=repeat_out,
            gene_model_path=gene_model_path,
            repeat_checkpoint_path=repeat_checkpoint_path,
            gene_checkpoint_format=gene_checkpoint_format,
            repeat_checkpoint_format=repeat_checkpoint_format,
            gene_config_path=gene_config_path,
            repeat_config_path=repeat_config_path,
            repeat_classifier_path=repeat_classifier_path,
            spe_embedding_path=spe_embedding_path,
            seq_len=seq_len,
            flank_bp=flank_bp,
            batch_size=batch_size,
            min_seq_len=min_seq_len,
            use_hmm=use_hmm,
            use_spe_embeddings=use_spe_embeddings,
            species_name=species_name,
            parallel_factor=parallel_factor,
            strand=strand,
            repeat_min_run_length=args.repeat_min_run_length,
            repeat_max_gap=args.repeat_max_gap,
        )
    finally:
        cleanup_runtime(runtime)

    if runtime.is_main:
        duration_minutes = (time.time() - start_time) / 60
        print(f"OrionGeno finished in {duration_minutes:.4f} minutes.")
