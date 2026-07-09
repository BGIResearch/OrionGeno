# Advancing ab initio gene annotation with OrionGeno

<p align="center">
  <strong>OrionGeno takes a genome FASTA and produces gene annotations in GTF format.</strong>
</p>

<p align="center">
  <a href="https://db.cngb.org/genomics/orion_geno"><img alt="Online API" src="https://img.shields.io/badge/Online%20API-CNGBdb-0ea5e9?style=flat-square&logo=fastapi&logoColor=white"></a>
  <a href="https://huggingface.co/BGI-Research/OrionGeno"><img alt="Hugging Face model weights" src="https://img.shields.io/badge/Model%20Weights-Hugging%20Face-ffcc4d?style=flat-square&logo=huggingface&logoColor=black"></a>
  <a href="https://modelscope.cn/models/BGI-Research/OrionGeno/files"><img alt="ModelScope model weights" src="https://img.shields.io/badge/ModelScope-Weights-6246ea?style=flat-square"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="Outputs" src="https://img.shields.io/badge/Outputs-Gene%20GTF%20%7C%20Repeat%20GTF-0a7f5a?style=flat-square">
  <img alt="License" src="https://img.shields.io/badge/License-Non--Commercial-b45309?style=flat-square">
</p>

<p align="center">
  <img src="docs/img/Fig2.png" alt="Protein benchmark summary" width="980">
</p>

<p align="center">
  <sub>Figure 2. Mean protein-level F1 across the tested models at coverage thresholds from 0.6 to 1.0. <a href="docs/img/Fig2.png">HD</a></sub>
</p>

## What OrionGeno Does

OrionGeno is a deep learning-based *ab initio* model for eukaryotic gene annotation. It takes FASTA-formatted genomic sequences as input and generates GTF annotations for gene structures, including exons, introns, UTRs, and optional repeat regions. Powered by a phylogeny-aware architecture, it achieves state-of-the-art accuracy across diverse lineages, including Vertebrates, Invertebrates, Viridiplantae, and Fungi.

## Online API

An online OrionGeno API is available at [https://db.cngb.org/genomics/orion_geno](https://db.cngb.org/genomics/orion_geno). We welcome researchers to test the service and share feedback. The API will be updated continuously as additional curated species annotations become available.

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

Clone the repository first:

```bash
git clone https://github.com/BGIResearch/OrionGeno.git
cd OrionGeno
```

Then follow the validated installation path in
[docs/oriongeno-installation-success-notes.md](docs/oriongeno-installation-success-notes.md).
That guide keeps package caches, temporary files, native extension builds, and
downloaded wheels inside the repository directory, and documents the tested
Python 3.10, PyTorch 2.7.0, CUDA 12.x, `mamba-ssm`, and `causal-conv1d`
installation path.

#### Docker

A `Dockerfile` is provided as a ready-to-run alternative to the manual
install above, using the same validated PyTorch, `mamba-ssm`/`causal-conv1d`,
and `numba` versions. Model weights are **not** baked into the image;
download them from Hugging Face or ModelScope (see Download Model Weights
below) and mount the checkpoint directory at run time.

```bash
docker build -t oriongeno .
```

If the default PyPI/Debian mirrors are slow for you, pass a regional mirror
at build time:

```bash
docker build \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/ \
  --build-arg APT_MIRROR=mirrors.tuna.tsinghua.edu.cn \
  -t oriongeno .
```

Run gene prediction on one GPU:

```bash
docker run --rm --gpus all \
  -v /path/to/oriongeno-weights/oriongeno_mammals:/weights:ro \
  -v /path/to/genome.fna:/data/genome.fna:ro \
  -v /path/to/output:/data/out \
  oriongeno \
  --genome /data/genome.fna \
  --output /data/out/oriongeno.gtf \
  --checkpoint /weights
```

Run gene prediction on multiple GPUs:

```bash
docker run --rm --gpus all \
  -v /path/to/oriongeno-weights/oriongeno_mammals:/weights:ro \
  -v /path/to/genome.fna:/data/genome.fna:ro \
  -v /path/to/output:/data/out \
  oriongeno \
  multi \
  --genome /data/genome.fna \
  --output /data/out/oriongeno.gtf \
  --checkpoint /weights \
  --devices 0,1,2,3
```

This requires a host with an NVIDIA GPU, a working NVIDIA driver, and Docker
configured for GPU access (`docker run --gpus all ...` working).

## Download Model Weights

Download the inference checkpoints from either Hugging Face or ModelScope via the links below:

| Model       | Hugging Face                                                            | ModelScope                                                                          |
|:-----------:|:----------------------------------------------------------------------- |:----------------------------------------------------------------------------------- |
| `OrionGeno` | [BGI-Research/OrionGeno](https://huggingface.co/BGI-Research/OrionGeno) | [BGI-Research/OrionGeno](https://modelscope.cn/models/BGI-Research/OrionGeno/files) |

For example, with Hugging Face:

```bash
git lfs install
git clone https://huggingface.co/BGI-Research/OrionGeno /path/to/oriongeno-weights
```

Or with ModelScope:

```bash
modelscope download --model BGI-Research/OrionGeno --local_dir /path/to/oriongeno-weights
```

To improve gene structure prediction accuracy across diverse species, OrionGeno provides lineage-specific checkpoint directories for groups such as mammals, birds, fish, other vertebrates, arthropods, other invertebrates, plants, and fungi. Download the checkpoint directory that matches your target lineage and pass that exact directory with `--checkpoint`.

For example:

```text
/path/to/oriongeno-weights/oriongeno_mammals/
```

## Quick Start

### Inputs

Three filesystem paths are required:

- checkpoint directory downloaded from Hugging Face or ModelScope
- genome FASTA input
- output GTF path

Specifying `--species-name` is optional. It is used only for species conditioning inside the selected checkpoint; it does not select or validate a checkpoint directory.

### Execution

Run gene prediction on one GPU:

```bash
python main.py \
  --genome /path/to/genome.fna \
  --output /path/to/output/oriongeno.gtf \
  --checkpoint /path/to/oriongeno-weights/oriongeno_mammals \
  --length 512000 \
  --flank 64000 \
  --batch-size 8 \
  --output-gene True \
  --output-repeat False \
  --species-name Homo_sapiens
```

Run gene prediction on multiple GPUs:

```bash
python main.py multi \
  --genome /path/to/genome.fna \
  --output /path/to/output/oriongeno.gtf \
  --checkpoint /path/to/oriongeno-weights/oriongeno_mammals \
  --devices 0,1,2,3 \
  --length 512000 \
  --flank 64000 \
  --batch-size 8 \
  --output-gene True \
  --output-repeat False \
  --species-name Homo_sapiens
```

Run without species conditioning:

```bash
python main.py \
  --genome /path/to/genome.fna \
  --output /path/to/output/oriongeno.gtf \
  --checkpoint /path/to/oriongeno-weights/oriongeno_mammals \
  --length 512000 \
  --flank 64000 \
  --batch-size 8 \
  --output-gene True \
  --output-repeat False
```

Recommended batch sizes:

| GPU      | Recommended batch size |
|:-------- |:---------------------- |
| RTX 4090 | `4`                    |
| A40      | `8`                    |
| A100     | `16`                   |

If CUDA memory is insufficient, lower `--batch-size`.

### Outputs

OrionGeno generates annotations for genes and optional repeats. At least one of `--output-gene` or `--output-repeat` must be set to `True`.

Gene annotations are written directly to the file path specified by `--output` when `--output-gene True`:

```text
/path/to/output/oriongeno.gtf
```

The gene GTF may include:

- `gene`
- `transcript`
- `exon`
- `intron`
- `CDS`
- `start_codon`
- `stop_codon`
- `five_prime_utr`
- `three_prime_utr`

Repeat annotations are generated beside the gene annotation when `--output-repeat True`. If `--output` is `/path/to/output/oriongeno.gtf`, the repeat output is:

```text
/path/to/output/oriongeno.repeat.gtf
```

## Arguments

| Argument                | Default                            | Meaning                                                                                                                            |
|:----------------------- |:---------------------------------- |:---------------------------------------------------------------------------------------------------------------------------------- |
| `--genome`              | `ORIONGENO_GENOME` or required     | Genome FASTA input path.                                                                                                           |
| `--output`              | `ORIONGENO_OUT` or required        | Gene GTF output path. Repeat output uses the same basename with `.repeat` inserted before the extension.                           |
| `--checkpoint`          | `ORIONGENO_CHECKPOINT` or required | OrionGeno checkpoint directory to use for inference.                                                                               |
| `--length`              | `512000`                           | Output window length in bases.                                                                                                     |
| `--flank`               | `0`                                | Context bases added to each side of every output window. With the default `0`, OrionGeno uses boundary re-prediction where needed. |
| `--batch-size`          | `8`                                | Number of sequence windows processed at once. Use the GPU recommendations above.                                                   |
| `--hmm-parallel-factor` | `0`                                | Override the HMM chunk-parallel factor. `0` chooses it automatically.                                                              |
| `--profile-hmm`         | `False`                            | Print per-batch HMM timing breakdowns.                                                                                             |
| `--output-gene`         | `True`                             | Write gene annotations.                                                                                                            |
| `--output-repeat`       | `False`                            | Optional repeat-region output.                                                                                                     |
| `--species-name`        | empty                              | Optional species name used for species conditioning only. It does not select a checkpoint.                                         |

Multi-GPU mode also accepts:

| Argument          | Default                                                    | Meaning                                                                                                    |
|:----------------- |:---------------------------------------------------------- |:---------------------------------------------------------------------------------------------------------- |
| `--devices`       | `ORIONGENO_DEVICES`, then `CUDA_VISIBLE_DEVICES`, then `0` | Comma-separated GPU IDs, for example `0,1,2,3`.                                                            |
| `--work-dir`      | `<output basename>.records`                                | Temporary directory for per-GPU FASTA, GTF, logs, and staged merged outputs.                               |
| `--keep-work-dir` | off                                                        | Keep the temporary work directory after a successful run. Failed runs keep it automatically for debugging. |

## Acceleration and Memory Optimization

### Fragmented Assemblies and CPU Memory

For highly fragmented assemblies, running every short scaffold as an independent
inference record would create heavy scheduling overhead. In `auto` mode,
OrionGeno can pack short scaffolds into pseudo-contigs and then remap predictions
back to the original FASTA records after inference. This improves throughput on
draft assemblies with many scaffolds.

CPU memory peaks are mainly controlled by how much sequence is prepared for a
prediction group at one time. To lower CPU memory usage, reduce the sequence
group target:

```bash
ORIONGENO_SEQUENCE_GROUP_SIZE=4000000 python main.py ...
```

If a fragmented assembly still creates very large packed records, reduce the
packed pseudo-contig target size as well:

```bash
ORIONGENO_PACK_TARGET_SIZE=10000000 python main.py ...
```

Smaller values reduce memory peaks but may increase the number of groups and
slightly reduce throughput.

### GPU Memory

For GPU memory pressure, reduce `--batch-size`:

```bash
python main.py ... --batch-size 2
```

The prediction runner also sets the following PyTorch allocator option unless it
is already configured:

```text
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Troubleshooting

If `mamba-ssm` or `causal-conv1d` fails to install, follow the validated wheel-based path in [docs/oriongeno-installation-success-notes.md](docs/oriongeno-installation-success-notes.md). Also confirm that the CUDA runtime, PyTorch build, and NVIDIA driver are compatible.

If prediction fails before inference starts, check that `--genome`, `--checkpoint`, and `--output` are set correctly, and that the checkpoint path points to the exact lineage-specific checkpoint directory you want to use.

## License

OrionGeno is distributed under the **OrionGeno Non-Commercial License (Academic & Non-Commercial Use Only)**.
The code and model weights in this repository are personal/academic research achievements.
To prevent unauthorized commercial exploitation, this project is released under a non-commercial license.
Commercial use requires a separate license.

1. **Academic & Non-Commercial Use**: You are free to use, modify, and distribute the software **strictly** for non-commercial, academic, and scientific research purposes, in compliance with this license.

2. **Commercial Use**: Any commercial utilization, including but not limited to selling the software, integrating it into commercial platforms, using it for enterprise internal production, or providing paid cloud services, is **strictly prohibited** without prior written permission and a separate commercial license from BGI-Research.

*For commercial licensing inquiries and formal cooperation, please [contact the development team](mailto:yinpeng@genomics.cn).*

## Citation

If you use this codebase, or otherwise find our work valuable, please cite OrionGeno:

Lin Liu, Xudong Cai, Shengfu Wang, Yuan Deng, Yiwen Wu, Youliang Pan, Jieyu Wang, Chao Zhang, Haopeng Xia, Nongzhang Tan, Kui Su, Yang Liu, Xuping Zhou, Longqi Liu, Tong Wei, Yong Zhang, Qiye Li, Yuxiang Li, Peng Yin, Xun Xu. Advancing ab initio genome annotation with OrionGeno. Preprint at bioRxiv https://doi.org/10.64898/2026.04.26.720859 (2026).

If you have any questions, please contact: yinpeng@genomics.cn
