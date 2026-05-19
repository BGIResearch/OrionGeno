# OrionGeno Parameter Guide

This guide documents the current prediction interface. OrionGeno now uses an
explicit checkpoint path; `--species-name` is optional and is used only for
species conditioning when the checkpoint supports it.

Removed from the public code path: species-to-checkpoint routing, CSV route
tables, coordinate-interval sharding, merge subcommands, and the HTTP API.

## Entry Points

Single GPU:

```bash
python main.py \
  --genome /path/to/genome.fna \
  --output /path/to/output/oriongeno.gtf \
  --checkpoint /path/to/checkpoint \
  --length 512000 \
  --flank 0 \
  --batch-size 8 \
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
  --devices 0,1,2,3 \
  --length 512000 \
  --flank 0 \
  --batch-size 8 \
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
| `--output` | `ORIONGENO_OUT` or empty | Gene GTF output path. Required unless the environment variable is set. |
| `--checkpoint` | `ORIONGENO_CHECKPOINT` or empty | OrionGeno checkpoint directory. Required unless the environment variable is set. |

## Prediction Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `--length` | `512000` | Central output window length in bases. |
| `--flank` | `0` | Context bases added to each side of every output window. With the default `0`, OrionGeno uses no-flank initial prediction and re-predicts disagreeing chunk boundaries with windows the same length as `--length`. |
| `--batch-size` | `8` | Number of windows processed per model batch. Reduce this if GPU memory is insufficient. |
| `--hmm-parallel-factor` | `0` | Override the HMM chunk-parallel factor. `0` keeps automatic selection. |
| `--profile-hmm` | `False` | Print per-batch HMM timing breakdowns. |
| `--output-gene` | `True` | Write gene annotation GTF to `--output`. |
| `--output-repeat` | `False` | Write repeat annotation GTF next to the gene output. |
| `--species-name` | `ORIONGENO_SPECIES_NAME` or empty | Optional species name for model conditioning only. |

At least one of `--output-gene` or `--output-repeat` must be `True`.

For each sequence group, positive and negative strand chunks are queued
together for model and HMM inference. The configured `--batch-size` still
controls each GPU batch, so this reduces scheduling overhead without increasing
the per-batch GPU window count.

The HMM time-parallel path is implemented in Python/PyTorch. When
`--hmm-parallel-factor` is greater than `1`, local Viterbi chunks are computed
in parallel, chunk borders are stitched, and full chunk paths are reconstructed
from compact backpointers. Set `ORIONGENO_HMM_PARALLEL_BACKPOINTERS=0` only to
debug against the older full-score parallel path.

## Multi-GPU Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `--devices` | `ORIONGENO_DEVICES`, then `CUDA_VISIBLE_DEVICES`, then `0` | Comma-separated GPU IDs controlled by the multi-GPU wrapper, for example `0,1,2,3`. |
| `--work-dir` | `<output basename>.records` | Temporary directory for per-shard FASTA, GTF, logs, and staged merged outputs. |
| `--keep-work-dir` | disabled | Keep the temporary directory after a successful run. Failed runs keep the work directory automatically for debugging and recovery. |

The multi-GPU wrapper splits by complete FASTA record only. It does not split
inside a chromosome, contig, or scaffold. If there are fewer non-empty FASTA
records than requested GPUs, empty shards are skipped.

For fragmented assemblies, global N-spacer packing is applied before GPU
sharding. The packed pseudo-contigs are then assigned to GPUs by whole record.
Predictions that cross an artificial N spacer are dropped from the packed-stage
merge, and the original scaffold(s) touched by those predictions are
re-predicted independently. Packed-stage predictions on those scaffold(s) are
then replaced by the scaffold-native re-prediction results.

Packed and recheck FASTA files are temporary staging inputs. After they are
split into per-GPU shard FASTA files, OrionGeno removes the staging FASTA by
default to reduce disk usage in large multi-species runs. Set
`ORIONGENO_KEEP_TEMP_FASTA=1` to keep them for debugging.

Merged multi-GPU IDs are prefixed internally to avoid collisions:

| Output stage | Prefix |
| --- | --- |
| Initial shard merge | `partN.` |
| Cross-N scaffold re-prediction merge | `recheckN.` |

These internal prefixes are only used in temporary work files. The final public
GTF is renumbered with clean IDs: gene/transcript IDs become `g1`, `g1.t1`,
`g2`, `g2.t1`, and repeat IDs become `r1`, `r2`, ...

## Outputs

Gene output is written exactly to `--output` when `--output-gene True`:

```text
/path/to/output/oriongeno.gtf
```

Repeat output is written beside it when `--output-repeat True`:

```text
/path/to/output/oriongeno.repeat.gtf
```

Multi-GPU merges are staged inside `--work-dir` first. Final output files are
replaced only after all initial shards, any cross-N re-prediction shards, and
final merges succeed. This prevents a failed re-prediction run from leaving a
partially replaced user-facing GTF.

## Fragmented Assemblies

The fragmented-assembly controls are intentionally environment variables rather
than public command-line options.

| Environment variable | Default | Description |
| --- | --- | --- |
| `ORIONGENO_ASSEMBLY_MODE` | `auto` | `auto`, `native`, or `packed`. |
| `ORIONGENO_FRAGMENTED_RECORD_THRESHOLD` | `1` | In `auto` mode, consider packing when the retained FASTA record count is above this value. |
| `ORIONGENO_PACK_SPACER_LEN` | `0` | Artificial `N` spacer length. `0` chooses `max(10000, flank)`. Positive values are still clamped to at least `flank`. |
| `ORIONGENO_PACK_TARGET_SIZE` | `20000000` | Approximate target size for each packed pseudo-contig. |

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

The default is `10240000`.

By default, positive and negative strands are processed separately to reduce CPU
memory peaks. On hosts with enough CPU RAM, set
`ORIONGENO_BATCH_BOTH_STRANDS=True` to queue both strands from the same sequence
group together, matching the original behavior. This can reduce scheduling
overhead and improve throughput, but it substantially increases CPU memory use.

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

For GPU memory pressure, reduce `--batch-size`. The prediction runner also sets
the following unless it is already configured:

```text
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Environment Variables

| Variable | Description |
| --- | --- |
| `ORIONGENO_GENOME` | Default genome FASTA path. |
| `ORIONGENO_OUT` | Default output GTF path. |
| `ORIONGENO_CHECKPOINT` | Default checkpoint directory. |
| `ORIONGENO_SPECIES_NAME` | Optional species conditioning name. |
| `ORIONGENO_DEVICES` | Default multi-GPU device list. |
| `CUDA_VISIBLE_DEVICES` | Fallback multi-GPU device list and per-process GPU visibility. |
| `ORIONGENO_KEEP_TEMP_FASTA` | Keep packed/recheck staging FASTA files after shard FASTA files are created; default removes them. |
| `ORIONGENO_SEQ_LEN` | Default output window length. |
| `ORIONGENO_FLANK_SIZE` | Default flank size. |
| `ORIONGENO_BATCH_SIZE` | Default batch size. |
| `ORIONGENO_HMM_PARALLEL_FACTOR` | Default HMM chunk-parallel override; `0` keeps automatic selection. |
| `ORIONGENO_HMM_PARALLEL_BACKPOINTERS` | Use compact Python/PyTorch backpointers for time-parallel HMM; default `1`. |
| `ORIONGENO_PROFILE_HMM` | Print HMM timing breakdowns when truthy. |
| `ORIONGENO_SPARSE_VITERBI` | Sparse fixed-topology Viterbi mode. Default `auto` keeps the current 20-state HMM on dense decoding; set `1` to force sparse or `0` to force dense. |
| `ORIONGENO_CUDA_CACHE_CLEAR_INTERVAL` | CUDA cache cleanup interval; default `0` disables per-batch cleanup. |
| `ORIONGENO_OUTPUT_GENE` | Default gene-output switch. |
| `ORIONGENO_OUTPUT_REPEAT` | Default repeat-output switch. |
| `ORIONGENO_SEQUENCE_GROUP_SIZE` | CPU-side prepared-model-base target per sequence group. |
| `ORIONGENO_BATCH_BOTH_STRANDS` | Batch positive and negative strands from the same sequence group together when truthy. Default `False` lowers CPU memory use; set `True` on high-memory CPU hosts when throughput is preferred. |
| `ORIONGENO_MAX_CHUNKS_PER_INFERENCE_GROUP` | Maximum model chunks processed per internal inference window. Default `256`; lower it to reduce CPU peak memory. |
| `ORIONGENO_ASSEMBLY_MODE` | Fragmented assembly mode. |
| `ORIONGENO_FRAGMENTED_RECORD_THRESHOLD` | Auto-packing record-count threshold. |
| `ORIONGENO_PACK_SPACER_LEN` | Artificial N spacer length for packed scaffolds. |
| `ORIONGENO_PACK_TARGET_SIZE` | Packed pseudo-contig target size. |
