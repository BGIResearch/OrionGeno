"""Conservative multi-GPU prediction by whole FASTA record."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

from .cli_config import env_bool, env_int, env_str, str_to_bool
from .constants import (
    DEFAULT_FRAGMENTED_RECORD_THRESHOLD,
    DEFAULT_PACK_SPACER_LEN,
    DEFAULT_PACK_TARGET_SIZE,
)
from .data_processing.fasta_io import count_fasta_records, load_genome
from .data_processing.fragmented import prepare_inference_genome
from .gtf_outputs import repeat_output_path
from .genome_anno import format_gtf_attributes, parse_gtf_attributes

KEEP_TEMP_FASTA = env_bool("ORIONGENO_KEEP_TEMP_FASTA", False)


def parse_device_list(value):
    devices = [item.strip() for item in str(value or "").split(",") if item.strip()]
    if not devices:
        raise ValueError("--devices must contain at least one GPU id.")
    return devices


def validate_multi_args(args):
    """Validate multi-GPU arguments and return the requested devices."""
    try:
        devices = parse_device_list(args.devices)
    except ValueError as error:
        logging.error(str(error))
        sys.exit(1)

    if not args.genome:
        logging.error("--genome or ORIONGENO_GENOME is required.")
        sys.exit(1)
    if not args.output:
        logging.error("--output or ORIONGENO_OUT is required.")
        sys.exit(1)
    if not args.checkpoint:
        logging.error("--checkpoint or ORIONGENO_CHECKPOINT is required.")
        sys.exit(1)
    if not args.output_gene and not args.output_repeat:
        logging.error("At least one of --output-gene or --output-repeat must be True.")
        sys.exit(1)
    return devices


def shard_dir_for_output(output_path):
    output_abs = os.path.abspath(output_path)
    root, _ = os.path.splitext(output_abs)
    return f"{root}.records"


def fasta_opener(path, mode="rt"):
    import bz2
    import gzip

    if str(path).endswith(".gz"):
        return gzip.open(path, mode)
    if str(path).endswith(".bz2"):
        return bz2.open(path, mode)
    return open(path, mode)


def iter_fasta_records(path):
    name = ""
    lines = []
    with fasta_opener(path, "rt") as file_obj:
        for line in file_obj:
            if line.startswith(">"):
                if name:
                    yield name, lines
                name = line[1:].strip().split()[0]
                lines = [line]
            else:
                lines.append(line)
    if name:
        yield name, lines


def write_fasta_record(file_obj, lines):
    for line in lines:
        file_obj.write(line)
        if not line.endswith("\n"):
            file_obj.write("\n")


def split_fasta_by_record(genome_path, shard_dir, num_shards):
    """Split FASTA by whole record using greedy base-count balancing."""
    os.makedirs(shard_dir, exist_ok=True)
    shard_paths = [os.path.join(shard_dir, f"input_{index}.fasta") for index in range(num_shards)]
    shard_handles = [open(path, "w", encoding="utf-8") for path in shard_paths]
    shard_bases = [0] * num_shards
    shard_records = [0] * num_shards

    try:
        for _, lines in iter_fasta_records(genome_path):
            bases = sum(len(line.strip()) for line in lines[1:])
            shard_index = min(range(num_shards), key=lambda index: (shard_bases[index], shard_records[index], index))
            write_fasta_record(shard_handles[shard_index], lines)
            shard_bases[shard_index] += bases
            shard_records[shard_index] += 1
    finally:
        for handle in shard_handles:
            handle.close()

    logging.info(
        "Split %s FASTA records across %s device shards.",
        sum(shard_records),
        num_shards,
    )
    for index, path in enumerate(shard_paths):
        logging.info(
            "Shard %s: records=%s bases=%s path=%s",
            index,
            shard_records[index],
            shard_bases[index],
            path,
        )
    return shard_paths


def write_fasta_records(genome, output_path, names=None):
    from Bio import SeqIO

    selected = list(genome.values()) if names is None else [genome[name] for name in names if name in genome]
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_obj:
        SeqIO.write(selected, file_obj, "fasta")
    return output_path


def remove_temp_fasta(path, reason):
    """Remove a generated FASTA once downstream shard FASTAs exist."""
    if KEEP_TEMP_FASTA:
        logging.info("Keeping temporary FASTA because ORIONGENO_KEEP_TEMP_FASTA is set: %s", path)
        return
    try:
        os.remove(path)
        logging.info("Removed temporary FASTA after %s: %s", reason, path)
    except FileNotFoundError:
        return


def prepare_multi_input(args, work_dir):
    assembly_mode = env_str("ORIONGENO_ASSEMBLY_MODE", "auto")
    fragmented_threshold = env_int(
        "ORIONGENO_FRAGMENTED_RECORD_THRESHOLD",
        DEFAULT_FRAGMENTED_RECORD_THRESHOLD,
    )
    if assembly_mode == "native":
        logging.info("Multi-GPU assembly mode: native")
        return os.path.abspath(args.genome), None, None

    record_count = count_fasta_records(args.genome)
    if assembly_mode == "auto" and record_count <= fragmented_threshold:
        logging.info(
            "Multi-GPU global packing skipped: records=%s threshold=%s.",
            record_count,
            fragmented_threshold,
        )
        return os.path.abspath(args.genome), None, None

    genome = load_genome(args.genome)
    inference_genome, coordinate_mapper = prepare_inference_genome(
        genome,
        assembly_mode=assembly_mode,
        seq_len=args.seq_len,
        flank_size=args.flank_size,
        min_seq_len=0,
        fragmented_record_threshold=fragmented_threshold,
        pack_threshold=0,
        pack_spacer_len=env_int("ORIONGENO_PACK_SPACER_LEN", DEFAULT_PACK_SPACER_LEN),
        pack_target_size=env_int("ORIONGENO_PACK_TARGET_SIZE", DEFAULT_PACK_TARGET_SIZE),
    )
    if not coordinate_mapper.has_packing:
        logging.info("Multi-GPU global packing not needed after genome inspection.")
        return os.path.abspath(args.genome), None, genome

    packed_fasta = os.path.join(work_dir, "packed_input.fasta")
    write_fasta_records(inference_genome, packed_fasta)
    logging.info(
        "Multi-GPU global packing enabled: %s input records -> %s inference records; packed FASTA=%s",
        coordinate_mapper.summary.get("kept_sequences"),
        coordinate_mapper.summary.get("inference_sequences"),
        packed_fasta,
    )
    return packed_fasta, coordinate_mapper, genome


def prefix_gtf_attributes(attributes, prefix):
    parsed = parse_gtf_attributes(attributes)
    extra = []
    gene_id = parsed.pop("gene_id", "")
    transcript_id = parsed.pop("transcript_id", "")
    if gene_id:
        gene_id = f"{prefix}.{gene_id}"
    if transcript_id:
        transcript_id = f"{prefix}.{transcript_id}"
    for key, value in parsed.items():
        extra.append(f'{key} "{value}";')
    return format_gtf_attributes(gene_id, transcript_id, extra)


def prefix_repeat_attributes(attributes, prefix):
    parsed = parse_gtf_attributes(attributes)
    repeat_id = parsed.pop("repeat_id", "")
    if repeat_id:
        parsed = {"repeat_id": f"{prefix}.{repeat_id}", **parsed}
    else:
        parsed = {"repeat_id": prefix, **parsed}
    return " ".join(f'{key} "{value}";' for key, value in parsed.items())


def normalize_gene_gtf_ids(input_path, output_path):
    """Write a user-facing gene GTF with clean sequential IDs.

    Multi-GPU merging uses shard prefixes such as ``part0.`` and ``recheck0.``
    internally so local IDs cannot collide. The final public GTF should match
    the single-GPU convention: ``g1``, ``g1.t1``, ``g2``, and so on.
    """
    gene_ids = OrderedDict()
    transcript_ids = OrderedDict()
    transcript_counts = {}

    with open(input_path, "r", encoding="utf-8") as in_obj:
        for line in in_obj:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                continue
            parsed = parse_gtf_attributes(fields[8])
            old_gene_id = parsed.get("gene_id", "")
            if not old_gene_id:
                continue
            if old_gene_id not in gene_ids:
                gene_ids[old_gene_id] = f"g{len(gene_ids) + 1}"
                transcript_ids[old_gene_id] = OrderedDict()
                transcript_counts[old_gene_id] = 0
            old_transcript_id = parsed.get("transcript_id", "")
            if old_transcript_id and old_transcript_id not in transcript_ids[old_gene_id]:
                transcript_counts[old_gene_id] += 1
                transcript_ids[old_gene_id][old_transcript_id] = (
                    f"{gene_ids[old_gene_id]}.t{transcript_counts[old_gene_id]}"
                )

    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(input_path, "r", encoding="utf-8") as in_obj, open(output_path, "w", encoding="utf-8") as out_obj:
        for line in in_obj:
            if not line.strip():
                out_obj.write(line)
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                out_obj.write(line)
                continue
            parsed = parse_gtf_attributes(fields[8])
            old_gene_id = parsed.pop("gene_id", "")
            old_transcript_id = parsed.pop("transcript_id", "")
            new_gene_id = gene_ids.get(old_gene_id, old_gene_id)
            new_transcript_id = old_transcript_id
            if old_transcript_id and old_gene_id in transcript_ids:
                new_transcript_id = transcript_ids[old_gene_id].get(old_transcript_id, old_transcript_id)
            extra = [f'{key} "{value}";' for key, value in parsed.items()]
            fields[8] = format_gtf_attributes(new_gene_id, new_transcript_id, extra)
            out_obj.write("\t".join(fields) + "\n")
    logging.info("Normalized final gene GTF IDs: %s genes -> %s", len(gene_ids), output_path)


def normalize_repeat_gtf_ids(input_path, output_path):
    """Write a user-facing repeat GTF with clean sequential repeat IDs."""
    repeat_ids = OrderedDict()
    with open(input_path, "r", encoding="utf-8") as in_obj:
        for line in in_obj:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                continue
            repeat_id = parse_gtf_attributes(fields[8]).get("repeat_id", "")
            if repeat_id and repeat_id not in repeat_ids:
                repeat_ids[repeat_id] = f"r{len(repeat_ids) + 1}"

    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(input_path, "r", encoding="utf-8") as in_obj, open(output_path, "w", encoding="utf-8") as out_obj:
        for line in in_obj:
            if not line.strip():
                out_obj.write(line)
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                out_obj.write(line)
                continue
            parsed = parse_gtf_attributes(fields[8])
            old_repeat_id = parsed.pop("repeat_id", "")
            if old_repeat_id:
                parsed = {"repeat_id": repeat_ids.get(old_repeat_id, old_repeat_id), **parsed}
            fields[8] = " ".join(f'{key} "{value}";' for key, value in parsed.items())
            out_obj.write("\t".join(fields) + "\n")
    logging.info("Normalized final repeat GTF IDs: %s repeats -> %s", len(repeat_ids), output_path)


def remap_gtf_fields(fields, coordinate_mapper):
    if coordinate_mapper is None or not coordinate_mapper.has_packing:
        return list(fields), set()
    sources = coordinate_mapper.source_names_for_interval(fields[0], fields[3], fields[4])
    mapped = coordinate_mapper.map_interval(fields[0], fields[3], fields[4])
    if mapped is None:
        return None, sources
    mapped_fields = list(fields)
    mapped_fields[0] = mapped.seq_name
    mapped_fields[3] = str(mapped.start)
    mapped_fields[4] = str(mapped.end)
    return mapped_fields, sources


def iter_gene_groups(input_path):
    groups = OrderedDict()
    with open(input_path, "r", encoding="utf-8") as in_obj:
        for line_number, line in enumerate(in_obj, start=1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                groups[f"__raw__{line_number}"] = [fields]
                continue
            parsed = parse_gtf_attributes(fields[8])
            gene_id = parsed.get("gene_id", f"__nogene__{line_number}")
            groups.setdefault(gene_id, []).append(fields)
    return groups.values()


def merge_gene_gtf_files(
    input_paths,
    output_path,
    coordinate_mapper=None,
    drop_sequences=None,
    prefix_base="part",
    append=False,
):
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    drop_sequences = set(drop_sequences or [])
    recheck_sequences = set()
    written = 0
    mode = "a" if append else "w"
    with open(output_path, mode, encoding="utf-8") as out_obj:
        for shard_index, input_path in enumerate(input_paths):
            if not os.path.exists(input_path):
                continue
            prefix = f"{prefix_base}{shard_index}"
            for group in iter_gene_groups(input_path):
                mapped_group = []
                touched_sources = set()
                mapped_seq_names = set()
                invalid = False
                for fields in group:
                    if len(fields) != 9:
                        mapped_group.append(fields)
                        continue
                    mapped_fields, sources = remap_gtf_fields(fields, coordinate_mapper)
                    touched_sources.update(sources)
                    if mapped_fields is None:
                        invalid = True
                        continue
                    mapped_seq_names.add(mapped_fields[0])
                    mapped_group.append(mapped_fields)

                if invalid or len(mapped_seq_names) > 1:
                    recheck_sequences.update(touched_sources)
                    continue
                if mapped_seq_names & drop_sequences:
                    continue

                for fields in mapped_group:
                    if len(fields) == 9:
                        fields[8] = prefix_gtf_attributes(fields[8], prefix)
                        out_obj.write("\t".join(fields) + "\n")
                    else:
                        out_obj.write("\t".join(fields) + "\n")
                    written += 1
    logging.info("Merged gene GTF records: %s -> %s", written, output_path)
    return recheck_sequences


def merge_repeat_gtf_files(
    input_paths,
    output_path,
    coordinate_mapper=None,
    drop_sequences=None,
    prefix_base="part",
    append=False,
):
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    drop_sequences = set(drop_sequences or [])
    recheck_sequences = set()
    written = 0
    mode = "a" if append else "w"
    with open(output_path, mode, encoding="utf-8") as out_obj:
        for shard_index, input_path in enumerate(input_paths):
            if not os.path.exists(input_path):
                continue
            prefix = f"{prefix_base}{shard_index}"
            with open(input_path, "r", encoding="utf-8") as in_obj:
                for line in in_obj:
                    if not line.strip():
                        continue
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) == 9:
                        mapped_fields, sources = remap_gtf_fields(fields, coordinate_mapper)
                        if mapped_fields is None:
                            recheck_sequences.update(sources)
                            continue
                        fields = mapped_fields
                        if fields[0] in drop_sequences:
                            continue
                        fields[8] = prefix_repeat_attributes(fields[8], prefix)
                        out_obj.write("\t".join(fields) + "\n")
                    else:
                        out_obj.write(line)
                    written += 1
    logging.info("Merged repeat GTF records: %s -> %s", written, output_path)
    return recheck_sequences


def build_shard_command(args, shard_genome, shard_output):
    """Build the single-GPU prediction command for one FASTA shard."""
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent.parent / "main.py"),
        "--genome",
        os.path.abspath(shard_genome),
        "--output",
        os.path.abspath(shard_output),
        "--checkpoint",
        os.path.abspath(args.checkpoint),
        "--length",
        str(args.seq_len),
        "--flank",
        str(args.flank_size),
        "--batch-size",
        str(args.batch_size),
        "--output-gene",
        str(args.output_gene),
        "--output-repeat",
        str(args.output_repeat),
    ]
    if getattr(args, "hmm_parallel_factor", 0):
        command.extend(["--hmm-parallel-factor", str(args.hmm_parallel_factor)])
    if getattr(args, "profile_hmm", False):
        command.extend(["--profile-hmm", str(args.profile_hmm)])
    if args.species_name:
        command.extend(["--species-name", args.species_name])
    return command


def launch_shard(args, device, shard_index, shard_genome, shard_output, shard_log, env_overrides=None):
    command = build_shard_command(args, shard_genome, shard_output)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    if env_overrides:
        env.update(env_overrides)
    log_obj = open(shard_log, "w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(Path(__file__).resolve().parent.parent),
        env=env,
        stdout=log_obj,
        stderr=subprocess.STDOUT,
        text=True,
    )
    logging.info("Launched shard %s on GPU %s; log=%s", shard_index, device, shard_log)
    return {"index": shard_index, "device": device, "process": process, "log_obj": log_obj, "log": shard_log}


def wait_for_shards(active):
    """Wait for shard subprocesses, stopping the run if any shard fails."""
    failed = []
    while active:
        time.sleep(1)
        for item in list(active):
            return_code = item["process"].poll()
            if return_code is None:
                continue
            item["log_obj"].close()
            active.remove(item)
            if return_code != 0:
                failed.append(item)
                logging.error(
                    "Shard %s failed on GPU %s with exit code %s. Log: %s",
                    item["index"],
                    item["device"],
                    return_code,
                    item["log"],
                )
            else:
                logging.info("Shard %s finished on GPU %s.", item["index"], item["device"])

        if failed:
            for item in active:
                item["process"].terminate()
            deadline = time.time() + 30
            for item in active:
                remaining = max(0, deadline - time.time())
                try:
                    item["process"].wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    item["process"].kill()
                    item["process"].wait()
                item["log_obj"].close()
            raise RuntimeError("One or more OrionGeno shard processes failed.")


def run_shards(args, devices, input_fasta, work_dir, output_prefix, remove_input_after_split=False):
    shard_dir = os.path.join(work_dir, output_prefix)
    shard_inputs = split_fasta_by_record(input_fasta, shard_dir, len(devices))
    shard_record_counts = [count_fasta_records(shard_input) for shard_input in shard_inputs]
    active_shards = [
        (index, devices[index], shard_input)
        for index, shard_input in enumerate(shard_inputs)
        if shard_record_counts[index] > 0
    ]
    for index, record_count in enumerate(shard_record_counts):
        if record_count == 0:
            logging.info("Shard %s has no FASTA records; GPU %s is skipped.", index, devices[index])
    if not active_shards:
        raise RuntimeError("No FASTA records available for multi-GPU prediction.")

    if remove_input_after_split:
        remove_temp_fasta(input_fasta, f"creating {output_prefix} shard FASTA files")

    env_overrides = {"ORIONGENO_ASSEMBLY_MODE": "native"}
    shard_outputs = [
        os.path.join(shard_dir, f"{output_prefix}_{index}.gtf")
        for index, _, _ in active_shards
    ]
    active = [
        launch_shard(
            args,
            device,
            index,
            shard_input,
            os.path.join(shard_dir, f"{output_prefix}_{index}.gtf"),
            os.path.join(shard_dir, f"{output_prefix}_{index}.log"),
            env_overrides=env_overrides,
        )
        for index, device, shard_input in active_shards
    ]
    wait_for_shards(active)
    return shard_outputs


def run_multi_prediction(args):
    devices = validate_multi_args(args)

    output_path = os.path.abspath(args.output)
    repeat_path = repeat_output_path(output_path)
    work_dir = os.path.abspath(args.work_dir or shard_dir_for_output(output_path))
    if os.path.exists(work_dir) and not args.keep_work_dir:
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    staged_output_path = os.path.join(work_dir, "merged.gtf")
    staged_repeat_path = repeat_output_path(staged_output_path)

    success = False
    try:
        input_fasta, coordinate_mapper, original_genome = prepare_multi_input(args, work_dir)
        shard_outputs = run_shards(
            args,
            devices,
            input_fasta,
            work_dir,
            "shard",
            remove_input_after_split=coordinate_mapper is not None and coordinate_mapper.has_packing,
        )

        recheck_sequences = set()
        if args.output_gene:
            recheck_sequences.update(
                merge_gene_gtf_files(
                    shard_outputs,
                    staged_output_path,
                    coordinate_mapper=coordinate_mapper,
                )
            )
        if args.output_repeat:
            recheck_sequences.update(
                merge_repeat_gtf_files(
                    [repeat_output_path(path) for path in shard_outputs],
                    staged_repeat_path,
                    coordinate_mapper=coordinate_mapper,
                )
            )

        if recheck_sequences and original_genome is not None:
            recheck_sequences = sorted(recheck_sequences)
            logging.info(
                "Re-predicting %s original scaffolds whose packed shard predictions crossed N spacers.",
                len(recheck_sequences),
            )
            retry_fasta = os.path.join(work_dir, "recheck_input.fasta")
            write_fasta_records(original_genome, retry_fasta, recheck_sequences)
            retry_outputs = run_shards(
                args,
                devices,
                retry_fasta,
                work_dir,
                "recheck",
                remove_input_after_split=True,
            )
            if args.output_gene:
                merge_gene_gtf_files(
                    shard_outputs,
                    staged_output_path,
                    coordinate_mapper=coordinate_mapper,
                    drop_sequences=recheck_sequences,
                )
                merge_gene_gtf_files(
                    retry_outputs,
                    staged_output_path,
                    prefix_base="recheck",
                    append=True,
                )
            if args.output_repeat:
                merge_repeat_gtf_files(
                    [repeat_output_path(path) for path in shard_outputs],
                    staged_repeat_path,
                    coordinate_mapper=coordinate_mapper,
                    drop_sequences=recheck_sequences,
                )
                merge_repeat_gtf_files(
                    [repeat_output_path(path) for path in retry_outputs],
                    staged_repeat_path,
                    prefix_base="recheck",
                    append=True,
                )
        elif recheck_sequences:
            logging.warning(
                "Detected cross-N shard predictions, but original genome records are unavailable for re-prediction."
            )

        if args.output_repeat:
            os.makedirs(os.path.dirname(repeat_path) or ".", exist_ok=True)
            normalized_repeat_path = os.path.join(work_dir, "merged.clean.repeat.gtf")
            normalize_repeat_gtf_ids(staged_repeat_path, normalized_repeat_path)
            os.replace(normalized_repeat_path, repeat_path)
        if args.output_gene:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            normalized_output_path = os.path.join(work_dir, "merged.clean.gtf")
            normalize_gene_gtf_ids(staged_output_path, normalized_output_path)
            os.replace(normalized_output_path, output_path)
        success = True
    finally:
        if success and not args.keep_work_dir and os.path.exists(work_dir):
            shutil.rmtree(work_dir)
        elif not success:
            logging.error("Multi-GPU run did not complete; keeping work directory for recovery: %s", work_dir)


def build_multi_parser(prog="main.py multi"):
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run OrionGeno on multiple GPUs by whole FASTA record.",
    )
    parser.add_argument("--genome", default=env_str("ORIONGENO_GENOME", ""))
    parser.add_argument("--output", default=env_str("ORIONGENO_OUT", ""))
    parser.add_argument("--checkpoint", default=env_str("ORIONGENO_CHECKPOINT", ""))
    parser.add_argument("--devices", default=env_str("ORIONGENO_DEVICES", env_str("CUDA_VISIBLE_DEVICES", "0")))
    parser.add_argument("--length", dest="seq_len", type=int, default=env_int("ORIONGENO_SEQ_LEN", 512000))
    parser.add_argument("--flank", dest="flank_size", type=int, default=env_int("ORIONGENO_FLANK_SIZE", 0))
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
    parser.add_argument("--output-gene", type=str_to_bool, default=env_bool("ORIONGENO_OUTPUT_GENE", True))
    parser.add_argument("--output-repeat", type=str_to_bool, default=env_bool("ORIONGENO_OUTPUT_REPEAT", False))
    parser.add_argument("--species-name", default=env_str("ORIONGENO_SPECIES_NAME", ""))
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--keep-work-dir", action="store_true")
    return parser
