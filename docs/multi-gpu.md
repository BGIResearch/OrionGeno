# Multi-GPU and multi-node prediction

## Multiple GPUs on one node

```bash
oriongeno multi \
  --genome genome.fna \
  --checkpoint model-weights/checkpoints/oriongeno_mammals \
  --output results/oriongeno.gff3 \
  --devices 0,1 \
  --batch-size auto \
  --hmm-decode-batch auto
```

Each GPU worker loads the checkpoint once. The scheduler assigns sequence
windows to those workers and assembles annotations in genomic order. This
allows a long chromosome to use more than one GPU.

`--window-task-size` controls the number of windows assigned per task.
`--work-dir` selects the temporary directory; `--keep-work-dir` retains it
after a successful run. Failed runs retain their temporary files.

## Multiple nodes

Launch one OrionGeno process per node. Each process controls the GPUs listed
in `--devices`. All nodes must see the same input files, checkpoint, output
path, and work directory.

For two nodes with two GPUs each, run this on node 0:

```bash
oriongeno multi-node \
  --genome /shared/genome.fna \
  --checkpoint /shared/model-weights/oriongeno_mammals \
  --output /shared/results/oriongeno.gtf \
  --work-dir /shared/work/oriongeno \
  --devices 0,1 \
  --num-nodes 2 --node-rank 0
```

Run the same command on node 1 with `--node-rank 1`. Multi-node prediction
uses FASTA-record shards and shared coordination files. Rank 0 combines the
completed GTF outputs.

The existing `oriongeno multi ... --num-nodes ... --node-rank ...` interface
also selects this scheduler. `--num-nodes auto --node-rank auto` reads supported
SLURM, OpenMPI, PMI, or torchrun environment values. The launcher must start
one OrionGeno process per node.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--num-nodes` | `1` | Total node count, or `auto`. |
| `--node-rank` | `0` | Zero-based node rank, or `auto`. |
| `--distributed-timeout` | `0` | Coordination wait limit in seconds; `0` waits indefinitely. |
| `--distributed-poll-interval` | `5` | Coordination polling interval in seconds. |
