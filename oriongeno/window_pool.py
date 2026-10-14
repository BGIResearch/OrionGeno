"""Persistent GPU workers for already planned windows, with ordered results.

Only numeric input windows and decoded labels cross the process boundary. The
coordinator retains scaffold coordinates, strand order and annotation ownership.
"""
import logging
import math
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from .inference_runtime import predict_with_memory_backoff


def worker_environment(worker_count, environ=None, cpu_count=None):
    """Divide default active CPU and repair-memory budgets across GPU workers."""
    env = dict(os.environ if environ is None else environ)
    # Window workers opt into deterministic convolutions; legacy CLI paths
    # keep their existing policy unless the user sets this environment flag.
    env.setdefault("ORIONGENO_DETERMINISTIC_CUDNN", "1")
    available = cpu_count or (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                              else (os.cpu_count() or 1))
    budget = max(1, int(env.get("ORIONGENO_CPU_BUDGET", available)))
    share = max(1, budget // max(1, worker_count))
    if not any(env.get(key) for key in ("ORIONGENO_TORCH_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")):
        env["ORIONGENO_TORCH_THREADS"] = str(min(2, max(1, share // 3)))
    if not any(env.get(key) for key in ("ORIONGENO_NUMBA_THREADS", "NUMBA_NUM_THREADS")):
        env["ORIONGENO_NUMBA_THREADS"] = str(min(4, max(1, share - 1)))
        env["NUMBA_NUM_THREADS"] = env["ORIONGENO_NUMBA_THREADS"]
    env.setdefault("ORIONGENO_HMM_REPAIR_WORKERS", str(min(4, max(1, share - 1))))
    total_memory = max(1, int(env.get("ORIONGENO_TOTAL_REPAIR_MEMORY_MB", 4096)))
    env.setdefault("ORIONGENO_HMM_REPAIR_MEMORY_MB", str(max(1, min(1024, total_memory // worker_count))))
    return env


def window_slices(count, workers, task_size=32):
    """Balance a short final wave instead of assigning one full tail job."""
    if count <= 0:
        return []
    waves = max(1, math.ceil(count / (max(1, workers) * max(1, task_size))))
    tasks = min(count, max(1, workers) * waves)
    return [(i * count // tasks, (i + 1) * count // tasks) for i in range(tasks)]


def _window_worker(connection, device, config, environment, log_path):
    os.environ.update(environment)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        # Import torch only after selecting the worker's visible device.
        from .inference_runtime import configure_inference_backend, configure_inference_threads
        from .prediction_annotator import PredictionAnnotator
        from .prediction_runner import _resolve_auto_batch_size, _resolve_auto_hmm_decode_batch
        import torch

        logging.basicConfig(filename=log_path, level=logging.INFO, force=True,
                            format="%(asctime)s %(message)s")
        configure_inference_threads()
        configure_inference_backend()
        if config.get("profile_hmm"):
            os.environ["ORIONGENO_PROFILE_HMM"] = "1"
        size = config["seq_len"] + 2 * config["flank_size"]
        batch = _resolve_auto_batch_size(config["batch_size"], size)
        decode_batch = _resolve_auto_hmm_decode_batch(config["hmm_decode_batch"], batch)
        predictor = PredictionAnnotator(
            model_path=config["checkpoint"], seq_len=config["seq_len"], batch_size=batch,
            hmm=config["use_hmm"], upper_only=config["upper_only"],
            species_name=config.get("species_name", ""),
            parallel_factor=config["parallel_factor"], hmm_decode_batch=decode_batch,
        )
        predictor.load_model(summary=False)
        connection.send({"ready": True, "batch_size": batch})
        while True:
            message = connection.recv()
            if message is None:
                break
            if message.get("operation") == "utr_repair":
                try:
                    with torch.inference_mode():
                        path, audit = predictor.repair_utr_region(
                            message["sequence"], message["original"], message["protected"], message["offset"],
                        )
                    connection.send({"id": message["id"], "path": path, "audit": audit})
                except (ValueError, MemoryError) as error:
                    connection.send({"id": message["id"], "repair_error": str(error)})
                continue
            started = time.perf_counter()
            chunks = message["chunks"]
            predictor.adapt_batch_size(chunks.shape[1])
            batch_limit = message.get("batch_size")
            if batch_limit:
                predictor.adapted_batch_size = min(predictor.adapted_batch_size, batch_limit)
            with torch.inference_mode():
                outputs = predict_with_memory_backoff(
                    predictor, chunks, output_gene=message["output_gene"],
                    output_repeat=message["output_repeat"],
                )
            outputs = {key: value.detach().cpu().numpy() if torch.is_tensor(value) else value
                       for key, value in outputs.items()}
            elapsed = time.perf_counter() - started
            logging.info("Task %s: windows=%d length=%d seconds=%.3f", message["id"],
                         len(chunks), chunks.shape[1], elapsed)
            connection.send({"id": message["id"], "outputs": outputs, "seconds": elapsed})
            predictor.sequence_predictions = None
            del outputs, chunks, message
    except BaseException:
        try:
            connection.send({"error": traceback.format_exc()})
        except (OSError, EOFError):
            pass
        raise
    finally:
        connection.close()


class WindowPool:
    """One bounded job per GPU, dynamic work assignment and deterministic assembly."""

    def __init__(self, devices, config, work_dir, task_size=32, worker_target=None):
        if not devices or len(set(devices)) != len(devices):
            raise ValueError("Window scheduling requires distinct GPU devices")
        if task_size < 1:
            raise ValueError("window-task-size must be positive")
        self.workers = []
        self.task_size = task_size
        self.counter = 0
        self.closed = False
        environment = worker_environment(len(devices))
        context = mp.get_context("spawn")
        try:
            for index, device in enumerate(devices):
                parent, child = context.Pipe()
                process = context.Process(
                    target=worker_target or _window_worker,
                    args=(child, device, config, environment,
                          str(Path(work_dir) / f"window_worker_{index}.log")),
                )
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    child.close()
                    raise
                child.close()
                self.workers.append({"device": device, "process": process, "connection": parent})
            ready = {}
            pending = list(self.workers)
            while pending:
                for worker in self._ready(pending):
                    message = self._receive(worker)
                    if not message.get("ready"):
                        raise RuntimeError("GPU worker did not send its initialization result")
                    ready[worker["device"]] = message["batch_size"]
                    worker["ready"] = True
                    pending.remove(worker)
            self.batch_size = min(ready.values())
        except BaseException:
            self.close()
            raise

    def _ready(self, workers):
        while True:
            connections = wait([worker["connection"] for worker in workers], timeout=1)
            if connections:
                return [worker for worker in workers if worker["connection"] in connections]
            for worker in workers:
                if not worker["process"].is_alive():
                    raise RuntimeError(f"GPU {worker['device']} worker exited unexpectedly")

    @staticmethod
    def _receive(worker):
        try:
            message = worker["connection"].recv()
        except (EOFError, OSError) as error:
            raise RuntimeError(f"Lost GPU {worker['device']} worker") from error
        if "error" in message:
            raise RuntimeError(f"GPU {worker['device']} worker failed:\n{message['error']}")
        return message

    def predict(self, chunks, save=False, batch_size=None, output_gene=True,
                output_repeat=False, encoding_layer_oracle=None):
        if self.closed:
            raise RuntimeError("Window pool is closed")
        if encoding_layer_oracle is not None:
            raise ValueError("Window workers accept sequence inputs, not oracle probabilities")
        if not output_gene and not output_repeat:
            return {}
        jobs = iter(window_slices(len(chunks), len(self.workers), self.task_size))
        active = []
        outputs = {}

        def submit(worker):
            bounds = next(jobs, None)
            if bounds is None:
                return False
            start, end = bounds
            self.counter += 1
            worker["job"] = (self.counter, start, end)
            worker["busy"] = True
            worker["connection"].send({
                "id": self.counter, "chunks": np.ascontiguousarray(chunks[start:end]),
                "output_gene": output_gene, "output_repeat": output_repeat,
                "batch_size": batch_size,
            })
            return True

        try:
            for worker in self.workers:
                if submit(worker):
                    active.append(worker)
            while active:
                for worker in self._ready(active):
                    message = self._receive(worker)
                    task_id, start, end = worker["job"]
                    if message.get("id") != task_id:
                        raise RuntimeError("Mismatched window task completion")
                    expected_heads = ({"gene"} if output_gene else set()) | ({"repeat"} if output_repeat else set())
                    if set(message["outputs"]) != expected_heads:
                        raise RuntimeError("Window worker returned unexpected output heads")
                    for key, value in message["outputs"].items():
                        if value.shape != (end - start, chunks.shape[1]):
                            raise RuntimeError("Window worker returned an invalid output shape")
                        if key not in outputs:
                            outputs[key] = np.empty(chunks.shape[:2], dtype=value.dtype)
                        outputs[key][start:end] = value
                    logging.info("GPU %s completed windows %d:%d in %.3fs", worker["device"],
                                 start, end, message["seconds"])
                    worker["busy"] = False
                    if not submit(worker):
                        active.remove(worker)
        except BaseException:
            self.close()
            raise
        return outputs

    def predict_gene(self, chunks, **kwargs):
        return self.predict(chunks, output_gene=True, output_repeat=False, **kwargs)["gene"]

    def repair_utr_region(self, sequence, original, protected, offset):
        """Run an occasional joint repair on a resident model after a window wave."""
        if self.closed or any(w.get("busy") for w in self.workers):
            raise RuntimeError("UTR repair requires an idle window pool")
        worker = self.workers[0]
        self.counter += 1
        task_id = self.counter
        worker["busy"] = True
        try:
            worker["connection"].send(dict(operation="utr_repair", id=task_id,
                sequence=sequence, original=original, protected=protected, offset=offset))
            self._ready([worker])
            message = self._receive(worker)
            if message.get("id") != task_id:
                raise RuntimeError("Mismatched UTR repair completion")
            worker["busy"] = False
            if "repair_error" in message:
                raise ValueError(message["repair_error"])
            return message["path"], message["audit"]
        except (ValueError, MemoryError):
            raise
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        # Idle workers exit normally so logs are flushed. A busy worker may be
        # blocked sending a result that the coordinator can no longer consume.
        for worker in self.workers:
            graceful = worker.get("ready") and not worker.get("busy")
            if graceful and worker["process"].is_alive():
                try:
                    worker["connection"].send(None)
                except (OSError, EOFError):
                    graceful = False
            worker["connection"].close()
            if not graceful and worker["process"].is_alive():
                worker["process"].terminate()
        deadline = time.monotonic() + 10
        for worker in self.workers:
            worker["process"].join(timeout=max(0, deadline - time.monotonic()))
            if worker["process"].is_alive():
                worker["process"].kill()
                worker["process"].join()


def run_window_prediction(args, devices):
    from .multi_predict import build_prediction_args, cleanup_work_dir, prepare_work_dir
    from .prediction_runner import (
        build_predict_parser, run_prediction, _load_runtime_env_config, _resolve_parallel_factor,
    )
    from . import prediction_runner as runner

    _load_runtime_env_config()
    single = build_predict_parser().parse_args(build_prediction_args(args))
    torch = sys.modules.get("torch")
    cuda_initialized = torch is not None and torch.cuda.is_initialized()
    if len(devices) == 1 and not cuda_initialized:
        # Select the physical GPU before the first CUDA call. A fresh CLI can
        # use the regular bounded forward/HMM pipeline with no worker or IPC.
        os.environ.update(worker_environment(1))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[0])
        logging.info("Single GPU %s: running inference directly without worker IPC.", devices[0])
        return run_prediction(single)
    if len(devices) == 1:
        # Embedded callers may already own a CUDA context. Changing their
        # visibility mask would not remap that context; isolate the worker.
        logging.info("CUDA is already initialized; using an isolated GPU worker.")
    config = vars(single).copy()
    config.update(use_hmm=runner.USE_HMM and single.output_gene, upper_only=runner.UPPER_ONLY,
                  parallel_factor=_resolve_parallel_factor(single, single.seq_len + 2 * single.flank_size,
                                                           single.seq_len))
    work_dir = prepare_work_dir(args.work_dir or args.output + ".windows", args.output, args.keep_work_dir)
    pool = None
    success = False
    try:
        pool = WindowPool(devices, config, work_dir, task_size=args.window_task_size)
        run_prediction(single, session={"window_pool": pool, "batch_size": pool.batch_size})
        success = True
    finally:
        if pool is not None:
            pool.close()
        if success and not args.keep_work_dir:
            cleanup_work_dir(work_dir, args.output)
        elif not success:
            logging.error("Window prediction failed; keeping logs: %s", work_dir)
