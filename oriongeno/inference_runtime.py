"""Conservative CPU defaults and optional cuDNN policy for CLI inference."""
import logging
import os


def predict_with_memory_backoff(predictor, chunks, **kwargs):
    """Retry a complete window group when forward/HMM overlap exhausts VRAM.

    This supplements the model's forward-batch and HMM's internal OOM retries.
    The smaller decode group persists for later work on this predictor.
    """
    import gc
    import sys

    kwargs.setdefault("save", False)
    while True:
        try:
            return predictor.get_prediction_outputs(chunks, **kwargs)
        except Exception as error:
            # Only a predictor using PyTorch can raise its CUDA OOM. Resolve
            # that already-loaded backend after failure so NumPy-only callers
            # do not acquire a PyTorch dependency just by using this wrapper.
            torch = sys.modules.get("torch")
            if torch is None or not isinstance(error, torch.cuda.OutOfMemoryError):
                raise
            span = predictor.hmm_decode_batch
            if not predictor.hmm or span <= 1:
                raise
            predictor.hmm_decode_batch = max(1, span // 2)
            predictor.sequence_predictions = None
            logging.warning("GPU inference hit CUDA OOM; retrying with HMM decode batch=%d",
                            predictor.hmm_decode_batch)
        # Leave the exception scope first so its traceback releases tensors.
        gc.collect()
        torch.cuda.empty_cache()


def configure_inference_backend():
    """Optionally select deterministic cuDNN convolution algorithms.

    An unset ORIONGENO_DETERMINISTIC_CUDNN retains the caller's backend policy.
    This controls cuDNN, not every third-party CUDA operator or GPU architecture.
    """
    requested = os.environ.get("ORIONGENO_DETERMINISTIC_CUDNN")
    if requested is None:
        return
    import torch

    enabled = requested.lower() not in {
        "0", "false", "no", "off",
    }
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = enabled
    logging.info("Inference cuDNN deterministic=%s", enabled)


def configure_inference_threads():
    import torch

    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    requested = os.environ.get("ORIONGENO_TORCH_THREADS")
    if requested is not None:
        torch.set_num_threads(max(1, int(requested)))
    elif not any(os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")):
        torch.set_num_threads(min(2, available))
    # Numba masks are thread-local. The pipeline worker installs this mask too.
    try:
        import numba
    except ImportError:
        logging.info("Inference CPU threads: torch=%d; Numba unavailable", torch.get_num_threads())
        return
    requested = os.environ.get("ORIONGENO_NUMBA_THREADS", os.environ.get("NUMBA_NUM_THREADS"))
    count = int(requested) if requested is not None else min(4, available)
    numba.set_num_threads(max(1, min(count, numba.config.NUMBA_NUM_THREADS)))
    logging.info("Inference CPU threads: torch=%d numba=%d", torch.get_num_threads(), numba.get_num_threads())
