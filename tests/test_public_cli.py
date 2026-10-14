"""Regressions for the prediction-only public command-line interface."""

import importlib.util
import subprocess
import sys

import pytest

from oriongeno import cli


@pytest.mark.parametrize("command", [[], ["predict"], ["multi"], ["multi-node"]])
def test_prediction_help_without_loading_gpu_backend(command):
    result = subprocess.run(
        [sys.executable, "-m", "oriongeno", *command, "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--genome" in result.stdout and "--checkpoint" in result.stdout


def test_version(capsys):
    cli.main(["--version"])
    assert capsys.readouterr().out.strip() == "1.0.2"


@pytest.mark.parametrize("command", ["evidence", "external-evidence", "evidence-merge"])
def test_unsupported_commands_are_rejected(command):
    with pytest.raises(SystemExit, match="Unknown OrionGeno command"):
        cli.main([command, "--help"])


def test_external_components_are_not_packaged():
    for name in ["evidence_runner", "evidence_export", "merge", "merge_features", "evidence_tools"]:
        assert importlib.util.find_spec("oriongeno." + name) is None


def test_multi_uses_window_scheduler(monkeypatch):
    from oriongeno import multi_predict

    monkeypatch.delenv("ORIONGENO_NUM_NODES", raising=False)
    monkeypatch.delenv("ORIONGENO_NODE_RANK", raising=False)
    calls = []
    monkeypatch.setattr(multi_predict, "run_multi_prediction", lambda args: calls.append(args))
    cli.main(["multi", "--genome", "input.fa", "--checkpoint", "weights", "--output", "out.gtf"])
    assert len(calls) == 1


def test_existing_multi_node_flags_keep_distributed_dispatch(monkeypatch):
    from oriongeno import distributed_predict

    calls = []
    monkeypatch.setattr(distributed_predict, "run_multi_prediction", lambda args: calls.append(args))
    cli.main([
        "multi", "--genome", "input.fa", "--checkpoint", "weights", "--output", "out.gtf",
        "--num-nodes", "2", "--node-rank", "1",
    ])
    assert len(calls) == 1 and calls[0].num_nodes == "2" and calls[0].node_rank == "1"
