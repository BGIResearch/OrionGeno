# Advancing ab initio gene annotation with OrionGeno
<p align="center">
  <strong>OrionGeno takes a genome FASTA and produces gene annotations in GTF format.</strong>
</p>

<p align="center">
  <img src="docs/img/Fig2.png" alt="Protein benchmark summary" width="980">
</p>

<p align="center">
  <sub>Figure 2. Mean protein-level F1 across the tested models at coverage thresholds from 0.6 to 1.0. <a href="docs/img/Fig2.png">HD</a></sub>
</p>

## What OrionGeno Does

OrionGeno is an end-to-end model for ab initio eukaryotic gene annotation.
You provide a genome FASTA, an output GTF path, a downloaded checkpoint root,
and a species name. OrionGeno resolves the proper internal checkpoint from the
species metadata and writes one filtered gene GTF to the requested output path.

The public interface treats OrionGeno as one species-conditioned model. Users
do not choose `checkpoints1` to `checkpoints8` manually; checkpoint routing is
handled automatically from `model_packages/species_routes.csv`.

## Benchmark

- [Gene benchmark](docs/benchmark/gene_benchmark.md)
- [Protein benchmark](docs/benchmark/protein_benchmark.md)

## Installation

#### Environment

- Linux
- Python `>=3.10,<3.11`
- NVIDIA GPU
- NVIDIA driver with a working CUDA runtime

#### Create and Install

```bash
git clone https://github.com/BGIResearch/OrionGeno.git
cd OrionGeno

conda env create -f environment.min.yml
conda activate oriongeno
pip install -e .
```

## Download Model Weights

Download the inference checkpoints from Hugging Face:

| Model | Hugging Face |
|:-----:|:-------------|
| `OrionGeno` | [BGI-Research/OrionGeno](https://huggingface.co/BGI-Research/OrionGeno) |

For example:

```bash
git lfs install
git clone https://huggingface.co/BGI-Research/OrionGeno /path/to/oriongeno-checkpoints
```

The checkpoint root should contain the routed checkpoint directories directly:

```text
/path/to/oriongeno-checkpoints/
  checkpoints1/
  checkpoints2/
  checkpoints3/
  checkpoints4/
  checkpoints5/
  checkpoints6/
  checkpoints7/
  checkpoints8/
```

Pass this root directory to OrionGeno. Do not pass one subdirectory such as
`checkpoints1` manually; `--species_name` resolves the proper internal
checkpoint automatically.

## Choose Species

Choose `--species-name` from `model_packages/species_routes.csv`. The command
accepts values from the `ScientificName`, `Species`, `Subspecies`, or `TaxID`
columns. If your exact species is not listed, choose the closest related
species already provided in the table. This species value is what determines
which internal checkpoint is used.

Validate a species name before running inference:

```bash
oriongeno route \
  --species-name Homo_sapiens \
  --checkpoint-root /path/to/oriongeno-checkpoints
```

If you only want to check whether the species exists in the table before the
weights are downloaded, add `--allow-missing-checkpoint`.

## Quick Start

Only three filesystem paths are required:

- checkpoint root downloaded from Hugging Face
- genome FASTA input
- output GTF path

The species name is selected from `model_packages/species_routes.csv`.

```bash
CHECKPOINT_ROOT=/path/to/oriongeno-checkpoints
GENOME_FASTA=/path/to/genome.fna
OUTPUT_GTF=/path/to/out.gtf
SPECIES_NAME=Homo_sapiens

oriongeno pipeline \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --genome "$GENOME_FASTA" \
  --out "$OUTPUT_GTF" \
  --species_name "$SPECIES_NAME" \
  --batch_size 2
```

The gene annotation is written exactly to the path passed by `--out`:

```text
/path/to/out.gtf
```

The same interface also accepts the explicit long option names:

```bash
oriongeno pipeline \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --genome "$GENOME_FASTA" \
  --output "$OUTPUT_GTF" \
  --species-name "$SPECIES_NAME" \
  --batch-size 2
```

To write repeat regions together with gene annotations, explicitly enable
repeat output:

```bash
oriongeno pipeline \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --genome "$GENOME_FASTA" \
  --out "$OUTPUT_GTF" \
  --species_name "$SPECIES_NAME" \
  --batch_size 2 \
  --output-gene True \
  --output-repeat True
```

This writes:

```text
/path/to/out.gtf
/path/to/out.repeat.gtf
```

### Multi-GPU

```bash
oriongeno multi \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --species_name "$SPECIES_NAME" \
  --genome "$GENOME_FASTA" \
  --out "$OUTPUT_GTF" \
  --devices 0,1 \
  --batch_size 2
```

`oriongeno multi` splits FASTA records into shards, runs one shard per available
GPU process, merges the shard GTF files, and writes the final gene GTF exactly
to `--out`.

### Recommended Batch Sizes

Use these as starting values for `--batch-size`:

| GPU | Recommended batch size |
|:----|:-----------------------|
| RTX 4090 | `2` |
| A40 | `4` |
| A100 | `8` |

If CUDA memory is insufficient, lower `--batch-size`.

## Input and Output

### Input formats

- `*.fa`
- `*.fasta`
- `*.fna`
- gzip-compressed FASTA such as `*.fa.gz`, `*.fasta.gz`, and `*.fna.gz`

### Output

For the standard gene-annotation workflow, OrionGeno writes one filtered GTF:

```text
/path/to/out.gtf
```

The GTF may include:

- `gene`
- `transcript`
- `exon`
- `intron`
- `CDS`
- `start_codon`
- `stop_codon`
- `five_prime_utr`
- `three_prime_utr`

If repeat output is enabled with `--output-repeat True`, repeat regions are
written to the same basename with `.repeat.gtf`.

## Arguments

| Argument | Default | Meaning |
|:---------|:--------|:--------|
| `--checkpoint-root`, `--model` | `./checkpoints` or `ORIONGENO_CHECKPOINT_ROOT` | Root directory containing `checkpoints1` through `checkpoints8`; the species name selects the internal checkpoint automatically. |
| `--species-table` | `model_packages/species_routes.csv` | Species-to-checkpoint routing table. |
| `--species_name`, `--species-name` | required | Species name or TaxID from the route table, for example `Homo_sapiens` or `9606`. |
| `--genome` | required | Genome FASTA input path. |
| `--out`, `--output` | required for normal use | Gene GTF output path. |
| `--batch_size`, `--batch-size` | `8` | Number of sequence windows processed at once. Use the GPU recommendations above. |
| `--length` | `512000` | Output window length in bases. |
| `--flank` | `128000` | Context bases added to each side of every output window. |
| `--output-gene` | `True` | Write gene annotations. |
| `--output-repeat` | `False` | Optional repeat-region output; the standard public workflow leaves this off. |

Multi-GPU mode also accepts:

| Argument | Default | Meaning |
|:---------|:--------|:--------|
| `--devices` | `CUDA_VISIBLE_DEVICES` or `0` | Comma-separated GPU IDs, for example `0,1`. |
| `--num-shards` | number of devices | Total number of FASTA shards. |
| `--shard-dir` | `<output>.shards` | Temporary per-shard output directory. |
| `--keep-shards` | off | Keep temporary shard files after a successful merge. |

## Troubleshooting

If `mamba-ssm` or `causal-conv1d` fails to install, first confirm that the
CUDA runtime, PyTorch build, and NVIDIA driver are compatible. The
`environment.min.yml` file pins the tested Python 3.10 runtime set used for
inference.

If species routing fails, check that `--species-name` is present in
`model_packages/species_routes.csv` and that `--checkpoint-root` points to the
directory containing `checkpoints1` through `checkpoints8`.

## Citation

If you use this codebase, or otherwise find our work valuable, please cite OrionGeno:

```bibtex
@article{OrionGeno,
  title={Advancing ab initio gene annotation with OrionGeno},
  author={xxx1, xxx2, xxx3, et al},
  journal={xxxx},
  year={2026}
}
```

Please [Contact Us](mailto:yinpeng@genomics.cn?subject=Regarding%20OrionGeno%20Feedback) if you have any questions.
