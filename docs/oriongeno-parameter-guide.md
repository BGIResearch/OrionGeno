# OrionGeno Parameter Guide

This guide documents the current prediction interface. OrionGeno now uses an
explicit checkpoint path; `--species-name` is optional and is used only for
species conditioning when the checkpoint supports it.

`oriongeno`, `python -m oriongeno`, and `python main.py` provide the same
prediction interface. Use `--help` for the complete command reference.

## Entry Points

Single GPU:

```bash
python main.py \
  --genome /path/to/genome.fna \
  --output /path/to/output/oriongeno.gtf \
  --checkpoint /path/to/checkpoint \
  --length 640000 \
  --flank 40000 \
  --batch-size auto \
  --hmm-decode-batch auto \
  --output-gene True \
  --output-repeat False \
  --species-name Homo_sapiens
```

Multiple GPUs:

```bash
python main.py multi \
  --genome /path/to/genome.fna \
  --output /path/to/output/oriongeno.gtf \
  --checkpoint /path/to/checkpoint \
  --devices 0,1 \
  --length 640000 \
  --flank 40000 \
  --batch-size auto \
  --hmm-decode-batch auto \
  --output-gene True \
  --output-repeat False \
  --species-name Homo_sapiens
```

`--species-name` can be omitted. In that case the species embedding pkl is not
read. If it is provided, OrionGeno looks up the name in
`oriongeno/model_packages/species_embeddings_pca_formatted_numpy_float32.pkl` and passes
the embedding to the model. It never selects a checkpoint.

## Required Inputs

| Parameter | Default | Description |
| --- | --- | --- |
| `--genome` | `ORIONGENO_GENOME` or empty | Input genome FASTA. Required unless the environment variable is set. Plain, `.gz`, and `.bz2` FASTA are supported. |
| `--output` | `ORIONGENO_OUT` or empty | Gene GFF3 or GTF output path. Required unless the environment variable is set. |
| `--checkpoint` | `ORIONGENO_CHECKPOINT` or empty | OrionGeno checkpoint directory. Required unless the environment variable is set. |

## Prediction Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--genome` | required, or `ORIONGENO_GENOME` | Input genome FASTA. |
| `--checkpoint` | required, or `ORIONGENO_CHECKPOINT` | Directory containing the model configuration and weights. |
| `--output` | required, or `ORIONGENO_OUT` | `.gff`/`.gff3` for GFF3, `.gtf` for GTF; optional `.gz`. |
| `--length` | `640000` | Output window length in bases. |
| `--flank` | `40000` | Context added on each side of an output window. |
| `--batch-size` | `auto` | Neural-model batch size. An integer sets it explicitly. |
| `--hmm-decode-batch` | `auto` | HMM batch size; `0` reuses the neural-model batch size. |
| `--hmm-parallel-factor` | `0` | HMM chunk parallelism; `0` selects it automatically. |
| `--window-sizing` | `legacy` | `legacy`, `compact-short`, or `compact`. |
| `--gene-filter-mode` | `none` | `strict` applies coding-structure filtering to the requested output. |
| `--output-gene` | `True` | Write gene annotations. |
| `--output-repeat` | `False` | Write repeat annotations beside the gene output. |
| `--species-name` | empty | Optional species conditioning inside the selected checkpoint. |
| `--profile-hmm` | `False` | Report prediction and HMM timing. |

At least one of `--output-gene` or `--output-repeat` must be `True`.

`legacy` retains the padded inference layout. `compact-short` reduces padding
for short records; `compact` also shortens final windows. These modes change
model context and can change predictions.

Positive and negative strand chunks can share a sequence group. Neural model
batches and HMM decoding batches are sized independently; memory backoff keeps
the reduced size for subsequent work. CPU Viterbi uses the Numba decoder when
available. The Python/PyTorch time-parallel fallback retains compact
backpointers; `ORIONGENO_HMM_PARALLEL_BACKPOINTERS=0` selects its older
full-score path for debugging.

## Multi-GPU Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--devices` | `auto` | Comma-separated GPU IDs, or automatic device discovery. |
| `--window-task-size` | `32` | Sequence windows assigned to a worker per task. |
| `--work-dir` | derived from output | Temporary prediction directory. |
| `--keep-work-dir` | off | Retain temporary prediction files after success. |

Each GPU worker loads its checkpoint once and processes sequence-window
tasks. Windows from the same chromosome can use several GPUs, including when
the input contains fewer FASTA records than the GPU count. With one selected
GPU and an uninitialized CUDA context, inference runs directly in the process.

Fragmented assemblies can be packed into pseudo-contigs before prediction.
Predictions crossing an artificial N spacer are discarded, and affected
scaffolds are re-predicted independently before coordinates are restored.

The existing whole-record, shared-filesystem workflow is available through
`oriongeno multi-node`. It supports `--num-nodes`, `--node-rank`,
`--distributed-timeout`, and `--distributed-poll-interval`. The same arguments
on `oriongeno multi` select that workflow. See
[multi-GPU and multi-node execution](multi-gpu.md) for commands.

In the multi-node workflow, packed/recheck staging FASTA files are removed
after shard inputs are created. Set `ORIONGENO_KEEP_TEMP_FASTA=1` to retain
them. Temporary shard identifiers use `partN.` or `recheckN.` prefixes;
final ID normalization removes these prefixes from the merged output.

## Outputs

Gene GFF3 or GTF output is written exactly to `--output` when `--output-gene True`:

```text
/path/to/output/oriongeno.gtf
```

Repeat output is written beside it when `--output-repeat True`:

```text
/path/to/output/oriongeno.repeat.gtf
```

Multi-node merges are staged inside `--work-dir` first. Final output files are
replaced only after all initial shards, any cross-N re-prediction shards, and
final merges succeed. This prevents a failed re-prediction run from leaving a
partially replaced user-facing GTF.

Use `.gff` or `.gff3` for GFF3 output, or `.gtf` for GTF. An additional `.gz`
extension enables compression. For example, `results/oriongeno.gff3` produces
`results/oriongeno.repeat.gff3` when repeat output is enabled.

## Fragmented Assemblies

The fragmented-assembly controls are intentionally environment variables rather
than public command-line options.

| Environment variable | Default | Description |
| --- | --- | --- |
| `ORIONGENO_ASSEMBLY_MODE` | `auto` | `auto`, `native`, or `packed`. |
| `ORIONGENO_FRAGMENTED_RECORD_THRESHOLD` | `1` | In `auto` mode, consider packing when the retained FASTA record count is above this value. |
| `ORIONGENO_PACK_SPACER_LEN` | `4096` | Artificial `N` spacer length between packed records; must be non-negative. |
| `ORIONGENO_PACK_TARGET_SIZE` | `51200000` | Approximate target size for each packed pseudo-contig. |

Use the default `auto` mode for most draft assemblies. Set
`ORIONGENO_ASSEMBLY_MODE=native` only when you need to preserve every original
FASTA record as an independent inference record.

## Memory Controls

CPU memory peaks are mostly driven by how many bases are prepared per sequence
group before model batches are moved to the GPU. Short scaffolds are bucketed by
their adaptive chunk size, so many small scaffolds can share a group without
being estimated as full `--length` windows. The target is split across strands
when positive and negative strands are queued together. Lower this value on
hosts with limited CPU RAM:

```bash
ORIONGENO_SEQUENCE_GROUP_SIZE=4000000 python main.py ...
```

The default is `51200000`.

By default, positive and negative strands share a sequence group
(`ORIONGENO_BATCH_BOTH_STRANDS=True`). Set
`ORIONGENO_BATCH_BOTH_STRANDS=False` to process strands separately and reduce
CPU memory peaks, at the cost of additional scheduling overhead.

Long chromosomes can still produce hundreds of model chunks even when the
sequence group target is small. To cap that peak, OrionGeno splits each sequence
group into internal inference windows before model prediction and HMM decoding:

```bash
ORIONGENO_MAX_CHUNKS_PER_INFERENCE_GROUP=256 python main.py ...
```

The default is `256`. When `ORIONGENO_BATCH_BOTH_STRANDS=True`, the value is
shared across both strands, so the default runs up to about 128 chunks per
strand in each internal window. Lower the value further, for example `128`, on
CPU-memory constrained hosts. Set it to `0` only to restore the old unbounded
per-group behavior.

Use `--batch-size auto` and `--hmm-decode-batch auto` for automatic sizing.
For GPU memory pressure with explicit batch sizes, reduce the affected batch.
The prediction runner also sets the following unless it is already configured:

```text
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Environment Variables

| Variable | Description |
| --- | --- |
| `ORIONGENO_GENOME` | Default genome FASTA path. |
| `ORIONGENO_OUT` | Default output GFF3 or GTF path. |
| `ORIONGENO_CHECKPOINT` | Default checkpoint directory. |
| `ORIONGENO_SPECIES_NAME` | Optional species conditioning name. |
| `ORIONGENO_DEVICES` | Default multi-GPU device list. |
| `CUDA_VISIBLE_DEVICES` | Fallback multi-GPU device list and per-process GPU visibility. |
| `ORIONGENO_KEEP_TEMP_FASTA` | Keep packed/recheck staging FASTA files after shard FASTA files are created; default removes them. |
| `ORIONGENO_SEQ_LEN` | Default output window length. |
| `ORIONGENO_FLANK_SIZE` | Default flank size. |
| `ORIONGENO_BATCH_SIZE` | Neural-model batch size; default `auto`. |
| `ORIONGENO_HMM_DECODE_BATCH` | Independent HMM batch size; default `auto`; `0` reuses the neural batch size. |
| `ORIONGENO_GENE_FILTER_MODE` | `none` by default; `strict` applies coding-structure filtering. |
| `ORIONGENO_HMM_PARALLEL_FACTOR` | Default HMM chunk-parallel override; `0` keeps automatic selection. |
| `ORIONGENO_HMM_PARALLEL_BACKPOINTERS` | Use compact Python/PyTorch backpointers for time-parallel HMM; default `1`. |
| `ORIONGENO_PROFILE_HMM` | Print HMM timing breakdowns when truthy. |
| `ORIONGENO_SPARSE_VITERBI` | Sparse fixed-topology Viterbi mode. Default `auto` keeps the current 20-state HMM on dense decoding; set `1` to force sparse or `0` to force dense. |
| `ORIONGENO_CUDA_CACHE_CLEAR_INTERVAL` | CUDA cache cleanup interval; default `0` disables per-batch cleanup. |
| `ORIONGENO_OUTPUT_GENE` | Default gene-output switch. |
| `ORIONGENO_OUTPUT_REPEAT` | Default repeat-output switch. |
| `ORIONGENO_SEQUENCE_GROUP_SIZE` | CPU-side prepared-model-base target per sequence group. |
| `ORIONGENO_BATCH_BOTH_STRANDS` | Batch positive and negative strands from the same sequence group together when truthy. Default `True` shares the group across strands; set `False` to lower CPU memory use. |
| `ORIONGENO_MAX_CHUNKS_PER_INFERENCE_GROUP` | Maximum model chunks processed per internal inference window. Default `256`; lower it to reduce CPU peak memory. |
| `ORIONGENO_ASSEMBLY_MODE` | Fragmented assembly mode. |
| `ORIONGENO_FRAGMENTED_RECORD_THRESHOLD` | Auto-packing record-count threshold. |
| `ORIONGENO_PACK_SPACER_LEN` | Artificial N spacer length for packed scaffolds. |
| `ORIONGENO_PACK_TARGET_SIZE` | Packed pseudo-contig target size. |

## CPU and memory settings

| Environment variable | Behavior |
| --- | --- |
| `ORIONGENO_TORCH_THREADS` | Explicit PyTorch CPU thread count. Otherwise a conservative default is used when OpenMP/MKL counts are unset. |
| `ORIONGENO_NUMBA_THREADS` | Active CPU decoder thread count; bounded by the Numba thread pool. |
| `NUMBA_NUM_THREADS` | Numba thread-pool limit. |
| `ORIONGENO_CUDA_CACHE_CLEAR_INTERVAL` | `0` disables periodic cache clearing; a positive value sets the interval. |
| `ORIONGENO_DETERMINISTIC_CUDNN` | Explicitly choose deterministic cuDNN convolution behavior. |

Automatic model and HMM batch sizing are independent. Memory backoff reduces
the affected batch and keeps the smaller value for subsequent work.
