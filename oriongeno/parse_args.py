import argparse
import os
from argparse import Namespace

from .genome_utils import DEFAULT_FLANK_BP, DEFAULT_SEQ_LEN, SEQGROUP_SIZE


def str2bool(value):
    """Parse common string forms of boolean CLI flags."""
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "t", "y", "yes"):
        return True
    if value in ("false", "0", "f", "n", "no"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def apply_internal_defaults(args: Namespace):
    """Populate the hidden runtime defaults shared by CLI and API entry points."""
    args.use_hmm = True
    args.use_spe_embeddings = True
    args.repeat_out = getattr(args, "repeat_out", "")
    args.repeat_min_run_length = getattr(args, "repeat_min_run_length", 1)
    args.repeat_max_gap = getattr(args, "repeat_max_gap", 30)
    args.sequence_level = getattr(args, "sequence_level", "all")
    args.sequence_name_include_regex = getattr(
        args,
        "sequence_name_include_regex",
        "",
    )
    args.sequence_name_exclude_regex = getattr(
        args,
        "sequence_name_exclude_regex",
        "",
    )
    args.scaffold_pack_mode = getattr(args, "scaffold_pack_mode", "auto")
    args.scaffold_pack_gap_bp = getattr(args, "scaffold_pack_gap_bp", 4096)
    args.scaffold_pack_target_bp = getattr(
        args,
        "scaffold_pack_target_bp",
        SEQGROUP_SIZE,
    )
    args.device = getattr(args, "device", "auto")
    args.dist_backend = getattr(args, "dist_backend", "")
    args.dist_tmp_dir = getattr(args, "dist_tmp_dir", "")
    args.dist_timeout_minutes = getattr(args, "dist_timeout_minutes", 120)
    args.local_rank = getattr(
        args,
        "local_rank",
        int(os.environ.get("LOCAL_RANK", -1)),
    )
    args.parallel_factor = getattr(args, "parallel_factor", 0)
    args.strand = getattr(args, "strand", "+,-")
    args.id_prefix = getattr(args, "id_prefix", "")
    args.min_genome_seqlen = getattr(args, "min_genome_seqlen", 0)
    args.model = getattr(args, "model", "")
    args.model_root = getattr(args, "model_root", "")
    args.species_table_path = getattr(args, "species_table_path", "")
    args.species_embedding_path = getattr(args, "species_embedding_path", "")
    return args


def parse_cmd():
    """Parse command-line arguments for the OrienGeno entry point."""
    main_parser = argparse.ArgumentParser(
        prog="OrienGeno",
        description=(
            "OrienGeno performs gene annotation and repeat annotation from a genome FASTA.\n\n"
            "Typical usage:\n"
            "  Gene only:   python main.py --genome genome.fa --species_name Homo_sapiens --out result.gtf\n"
            "  Repeat only: python main.py --genome genome.fa --species_name Homo_sapiens --out result.gtf --use_gene_annotation False --use_repeat_annotation True\n"
            "  Both:        python main.py --genome genome.fa --species_name Homo_sapiens --out result.gtf --use_repeat_annotation True\n\n"
            "The public CLI only exposes the common user-facing options."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    subparser = main_parser.add_subparsers(
        dest="command",
        required=True
    )

    parser = subparser.add_parser(
        'pipeline'
    )
    parser.add_argument(
        "--model",
        type=str,
        help=argparse.SUPPRESS,
        default="",
    )
    parser.add_argument(
        "--model_root",
        type=str,
        help=argparse.SUPPRESS,
        default="",
    )
    parser.add_argument(
        "--species_table_path",
        type=str,
        help=argparse.SUPPRESS,
        default="",
    )
    parser.add_argument(
        "--species_embedding_path",
        type=str,
        help=argparse.SUPPRESS,
        default="",
    )
    parser.add_argument(
        "--out",
        type=str,
        help="Base output GTF path. Gene output always uses *.gene.gtf and repeat output always uses *.repeat.gtf.",
        default="oriengeno.gtf",
    )
    parser.add_argument(
        "--use_gene_annotation",
        type=str2bool,
        default=True,
        help="Whether to run gene annotation and write the gene GTF. Defaults to True.",
    )
    parser.add_argument(
        "--use_repeat_annotation",
        type=str2bool,
        default=False,
        help="Whether to run repeat annotation and write the repeat GTF. Defaults to False.",
    )
    parser.add_argument(
        "--genome",
        type=str,
        required=True,
        help="Genome sequence file in FASTA format.",
    )
    parser.add_argument(
        "--include_utr",
        type=str2bool,
        default=False,
        help="Write UTR rows to the output GTF using five_prime_utr/three_prime_utr feature names. Defaults to False.",
    )
    parser.add_argument(
        "--species_name",
        type=str,
        required=True,
        help='Required species name for the unified OrionGeno model, for example "Homo_sapiens" or "Homo sapiens".',
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        help="Core output length per inference tile. When flank_bp > 0, the model sees seq_len + 2 * flank_bp bases. Defaults to 512000.",
        default=DEFAULT_SEQ_LEN,
    )
    parser.add_argument(
        "--flank_bp",
        type=int,
        help="Extra context added on each side of every inference tile. The model sees seq_len + 2 * flank_bp bases, but only the center seq_len bases are written to output. Defaults to 128000.",
        default=DEFAULT_FLANK_BP,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Number of sub-sequences per batch.",
        default=2,
    )

    args = main_parser.parse_args()
    return apply_internal_defaults(args)
