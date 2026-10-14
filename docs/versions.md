# Version history

## OrionGeno V1.0.2 — 2026-10-14

This version updates the sequence-prediction engine, with inference speed and
resource use as the main focus.

| Area | Changes |
| --- | --- |
| Multi-GPU scheduling | Persistent GPU workers load their checkpoint once and process balanced sequence-window tasks. Long chromosomes can use multiple GPUs. |
| Batch sizing | Model-forward and HMM decode batches have separate automatic sizing and memory backoff. |
| Prediction and decoding | Model inference can overlap HMM decoding; existing outputs and reduced batch sizes are reused. |
| Model loading | Weights load directly with reduced CPU memory use; changing a batch size does not reload the checkpoint. |
| CPU/GPU resources | Conservative CPU thread defaults reduce contention. CUDA cache clearing is configurable rather than performed every batch. |
| Viterbi decoding | Parallel CPU decoding releases the GIL and normalizes cumulative scores to retain float32 precision. |
| Annotation assembly | Window stitching and UTR boundary repair preserve coding anchors across window boundaries. GFF3 and GTF outputs share the current prediction serializer. |
| Installation | The Python package and `oriongeno` command report version `1.0.2`; `main.py` remains supported. |
| Existing multi-node runs | Shared-filesystem coordination and existing node-rank arguments remain supported. |
| Benchmark | The current tables and four figures cover 20 species, including *Populus trichocarpa*, with runtime and checkpoints. Test hardware: 32 CPU threads + 2 × NVIDIA RTX 4090 D GPUs. |

See [parameters](oriongeno-parameter-guide.md),
[multi-GPU execution](multi-gpu.md), and [benchmark results](benchmark.md).

## OrionGeno V1.0.1

The previous public version is recorded at
[commit 37559f6](https://github.com/BGIResearch/OrionGeno/tree/37559f68e9ca84568757b5ffd059086d6b4540e6).

- Genome-sequence prediction with lineage-specific checkpoints and optional
  species conditioning.
- Gene GTF output and optional repeat annotations.
- Prediction distribution by FASTA records, with shared-filesystem multi-node
  coordination.
- Packing of fragmented assemblies and CPU Viterbi decoding.
- Linux GPU installation instructions, Docker support, and gene/protein benchmark
  tables.

[V1.0.1 benchmark archive](benchmark/README.md).
