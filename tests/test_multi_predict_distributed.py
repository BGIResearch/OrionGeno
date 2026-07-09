import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from oriongeno import multi_predict


class DistributedMultiPredictTests(unittest.TestCase):
    def write_fasta(self, path, lengths):
        with open(path, "w", encoding="utf-8") as file_obj:
            for index, length in enumerate(lengths):
                file_obj.write(f">seq{index}\n")
                file_obj.write("A" * length + "\n")

    def test_stage_manifest_assigns_contiguous_node_shards(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fasta_path = os.path.join(temp_dir, "input.fasta")
            work_dir = os.path.join(temp_dir, "work")
            self.write_fasta(fasta_path, [10, 20, 30, 40, 50])

            manifest = multi_predict.create_stage_manifest(
                fasta_path,
                work_dir,
                "shard",
                total_shards=4,
                num_nodes=2,
                devices_per_node=2,
            )

            self.assertEqual(manifest["total_shards"], 4)
            self.assertEqual(sum(manifest["shard_record_counts"]), 5)
            self.assertEqual(len(multi_predict.manifest_output_paths(manifest)), 4)

            node_one = multi_predict.assigned_stage_shards(manifest, ["0", "1"], node_rank=1)
            self.assertEqual([2, 3], [assignment["shard_index"] for assignment in node_one])
            self.assertEqual(["0", "1"], [assignment["device"] for assignment in node_one])
            for assignment in node_one:
                self.assertTrue(os.path.exists(assignment["input"]))

    def test_empty_shards_are_skipped_for_a_node(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fasta_path = os.path.join(temp_dir, "input.fasta")
            work_dir = os.path.join(temp_dir, "work")
            self.write_fasta(fasta_path, [10, 20])

            manifest = multi_predict.create_stage_manifest(
                fasta_path,
                work_dir,
                "shard",
                total_shards=4,
                num_nodes=2,
                devices_per_node=2,
            )

            node_one = multi_predict.assigned_stage_shards(manifest, ["0", "1"], node_rank=1)
            self.assertEqual([], node_one)
            self.assertEqual(len(multi_predict.manifest_output_paths(manifest)), 2)

    def test_device_count_mismatch_raises(self):
        manifest = {
            "stage": "shard",
            "devices_per_node": 2,
            "total_shards": 2,
            "shard_record_counts": [1, 1],
            "shard_inputs": ["input_0.fasta", "input_1.fasta"],
            "shard_outputs": ["shard_0.gtf", "shard_1.gtf"],
            "shard_logs": ["shard_0.log", "shard_1.log"],
        }

        with self.assertRaises(RuntimeError):
            multi_predict.assigned_stage_shards(manifest, ["0"], node_rank=0)

    def test_resolve_slurm_auto_rank(self):
        args = SimpleNamespace(num_nodes="auto", node_rank="auto")
        env = {
            "SLURM_JOB_NUM_NODES": "3",
            "SLURM_NODEID": "2",
            "SLURM_LOCALID": "0",
        }

        with patch.dict(os.environ, env, clear=True):
            num_nodes, node_rank = multi_predict.resolve_distributed_args(args)

        self.assertEqual(num_nodes, 3)
        self.assertEqual(node_rank, 2)
        self.assertEqual(args.num_nodes, 3)
        self.assertEqual(args.node_rank, 2)

    def test_resolve_torchrun_auto_rank(self):
        args = SimpleNamespace(num_nodes="auto", node_rank="auto")
        env = {
            "WORLD_SIZE": "2",
            "LOCAL_WORLD_SIZE": "1",
            "GROUP_RANK": "1",
            "RANK": "1",
        }

        with patch.dict(os.environ, env, clear=True):
            num_nodes, node_rank = multi_predict.resolve_distributed_args(args)

        self.assertEqual(num_nodes, 2)
        self.assertEqual(node_rank, 1)

    def test_auto_rank_rejects_multiple_launcher_tasks_per_node(self):
        args = SimpleNamespace(num_nodes="auto", node_rank="auto")
        env = {
            "WORLD_SIZE": "8",
            "LOCAL_WORLD_SIZE": "4",
            "GROUP_RANK": "1",
            "RANK": "4",
        }

        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError):
                multi_predict.resolve_distributed_args(args)


if __name__ == "__main__":
    unittest.main()
