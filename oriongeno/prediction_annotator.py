"""Prediction and annotation generation utilities for OrionGeno."""

import gc
import json
import logging
import os
import pickle
import time
from contextlib import nullcontext
import numpy as np
import torch
from tqdm import tqdm

from .constants import DEFAULT_SEQUENCE_LENGTH
from .data_processing.genome_fasta import GenomeSequences
from .genome_annotation import Anno

GENE_LABEL_COUNT = 20
REPEAT_LABEL_COUNT = 2
BOUNDARY_REPREDICTION_LENGTH_FACTOR = 2
MODEL_INFERENCE_DTYPE = torch.bfloat16
HMM_INFERENCE_DTYPE = torch.float32
INFERENCE_DTYPE = MODEL_INFERENCE_DTYPE
GENE_FEATURE_NAMES = ["intergenic", "intron", "CDS", "5UTR", "3UTR"]
DEFAULT_SPECIES_EMBEDDING_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "model_packages",
    "species_embeddings_pca_formatted_numpy_float32.pkl",
)


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


CUDA_CACHE_CLEAR_INTERVAL = max(0, _env_int("ORIONGENO_CUDA_CACHE_CLEAR_INTERVAL", 0))

# HMM Viterbi decode batch size, decoupled from the model-forward batch size.
# The forward batch is constrained by transformer activation memory, but the
# HMM decode only consumes the already-computed gene probabilities plus the
# nucleotides; its Viterbi tensors are far lighter, so many more chunks can be
# decoded per viterbi() call. Larger decode batches give the CPU decoder more
# batch-level work and amortize per-step CUDA kernel-launch overhead on the GPU
# fallback path.
# 0 = fall back to the model-forward batch size (legacy behavior).
HMM_DECODE_BATCH_SIZE = max(0, _env_int("ORIONGENO_HMM_DECODE_BATCH", 0))

GENE_20_TO_FEATURE = np.array(
    [
        0,  # IR
        1, 1, 1,  # coding intron phase 0/1/2
        2, 2, 2,  # coding exon phase 0/1/2
        2,  # START
        2, 2, 2,  # coding exon-to-intron
        2, 2, 2,  # coding intron-to-exon
        2,  # STOP
        3,  # 5UTR
        4,  # 3UTR
        1,  # UTR_EI
        1,  # UTR_INTRON
        1,  # UTR_IE
    ],
    dtype=np.int64,
)
HMM_PREFILTER_WINDOW = 200
HMM_PREFILTER_CDS_THRESHOLD = 0.8
CDS_LABEL_INDICES = np.flatnonzero(GENE_20_TO_FEATURE == 2)


class PredictionAnnotator:
    """Run OrionGeno inference and convert predictions into annotation records.

    Attributes:
        model_path (str): Path to the pre-trained model.
        seq_len (int): Length of the sequences to process.
        batch_size (int): Batch size for prediction.
        hmm (bool): Whether to use HMM Viterbi decoding.
        model_device (torch.device): Device used by the sequence model.
    """

    def __init__(
        self,
        model_path="",
        seq_len=DEFAULT_SEQUENCE_LENGTH,
        batch_size=200,
        hmm=False,
        temp_dir="",
        annot_path="",
        genome_path="",
        genome=None,
        upper_only=True,
        species_name="",
        strand="+",
        parallel_factor=1,
        hmm_decode_batch=0,
    ):
        """Configure the prediction runner.

        Args:
            model_path (str): Path to the 20-label OrionGeno checkpoint.
            seq_len (int): Central sequence length emitted per chunk.
            batch_size (int): Number of chunks processed per model batch.
            hmm (bool): Whether to apply HMM Viterbi decoding.
            temp_dir (str): Temporary directory for cached intermediate arrays.
            annot_path (str): Optional reference annotation path.
            genome_path (str): Input genome FASTA path.
            genome: Optional in-memory sequence-record mapping.
            upper_only (bool): Whether to ignore lowercase softmask tracks.
            strand (str): Strand to process, either "+" or "-".
            parallel_factor (int): Chunk-parallel factor used by Viterbi.
        """
        self.model_path = model_path
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.adapted_batch_size = batch_size
        self.annot_path = annot_path
        self.genome_path = genome_path
        self.genome = genome
        self.upper_only = upper_only
        self.species_name = species_name
        self.species_embedding_key = ""
        self.species_embedding_vector = None
        self.hmm = hmm
        self.strand = strand
        self.model = None
        self.fasta_seq_lens = {}
        if temp_dir and not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
        self.temp_dir = temp_dir
        self.sequence_predictions = None
        self.parallel_factor = parallel_factor
        # Explicit constructor arg wins; otherwise fall back to the env override.
        self.hmm_decode_batch = int(hmm_decode_batch) or HMM_DECODE_BATCH_SIZE
        self.sequence_model = None
        self.gene_label_count = GENE_LABEL_COUNT
        self.repeat_label_count = REPEAT_LABEL_COUNT
        self.hmm_label_count = GENE_LABEL_COUNT
        self.model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cuda_cache_clear_interval = CUDA_CACHE_CLEAR_INTERVAL
        self._cuda_cache_release_calls = 0

    def reduce_label(self, arr):
        """Reduce 20-label gene states to gene feature classes.

        Args:
            arr (np.ndarray): 20-label gene state ids.

        Returns:
            np.ndarray: Feature ids for intergenic, intron, CDS, 5UTR, and 3UTR.
        """
        arr = np.asarray(arr, dtype=np.int64)
        if arr.size == 0:
            return arr

        label_min = int(arr.min())
        label_max = int(arr.max())
        if label_min < 0 or label_max >= GENE_LABEL_COUNT:
            raise ValueError(
                f"Unsupported gene label range [{label_min}, {label_max}]; "
                f"expected 20-label ids in [0, {GENE_LABEL_COUNT - 1}]."
            )
        return GENE_20_TO_FEATURE[arr]

    def _chunk_has_cds_signal(self, chunk_probs):
        """Return True when a chunk has enough CDS probability to need HMM decoding."""
        chunk_probs = np.asarray(chunk_probs)
        if chunk_probs.ndim != 2 or chunk_probs.shape[-1] != GENE_LABEL_COUNT:
            raise ValueError(
                "HMM prefilter expects chunk probabilities with shape "
                f"(chunk_length, {GENE_LABEL_COUNT}), got {chunk_probs.shape}."
            )

        cds_prob = chunk_probs[:, CDS_LABEL_INDICES].sum(axis=-1)
        if cds_prob.size == 0:
            return False

        window = HMM_PREFILTER_WINDOW
        if cds_prob.size < window:
            return float(cds_prob.mean()) >= HMM_PREFILTER_CDS_THRESHOLD

        cumsum = np.cumsum(cds_prob, dtype=np.float32)
        window_sums = cumsum[window - 1:] - np.concatenate(([0.0], cumsum[:-window]))
        return float(window_sums.max()) / window >= HMM_PREFILTER_CDS_THRESHOLD

    def _read_model_config(self):
        config_path = os.path.join(self.model_path, "config.json")
        if not os.path.exists(config_path):
            return {}
        with open(config_path, "r", encoding="utf-8") as file_obj:
            return json.load(file_obj)

    def _load_sequence_model(self):
        config = self._read_model_config()
        architectures = config.get("architectures") or []
        model_type = config.get("model_type")

        if model_type not in (None, "orion_geno") and not any("OrionGeno" in name for name in architectures):
            raise ValueError(
                "Only 20-label OrionGeno three-head checkpoints are supported by this pipeline."
            )

        from .model_packages import OrionGenoForJointLabeling

        # low_cpu_mem_usage=True loads checkpoint weights straight into empty
        # tensors and skips the random _init_weights pass that is immediately
        # overwritten by checkpoint values. The nested fallback keeps both
        # dtype-vs-torch_dtype compatibility and older transformers support.
        def _from_pretrained(**extra):
            try:
                return OrionGenoForJointLabeling.from_pretrained(
                    self.model_path,
                    dtype=INFERENCE_DTYPE,
                    **extra,
                )
            except TypeError:
                return OrionGenoForJointLabeling.from_pretrained(
                    self.model_path,
                    torch_dtype=INFERENCE_DTYPE,
                    **extra,
                )

        try:
            model = _from_pretrained(low_cpu_mem_usage=True)
        except TypeError:
            model = _from_pretrained()
        settings = getattr(model.config, "settings", {}) or {}
        self.gene_label_count = int(settings.get("gene_label_count", model.gene_label_count))
        self.repeat_label_count = int(settings.get("repeat_label_count", model.repeat_label_count))
        if self.gene_label_count != GENE_LABEL_COUNT:
            raise ValueError(
                f"Expected a {GENE_LABEL_COUNT}-label gene head, got {self.gene_label_count}."
            )
        if self.repeat_label_count != REPEAT_LABEL_COUNT:
            raise ValueError(
                f"Expected a {REPEAT_LABEL_COUNT}-label repeat head, got {self.repeat_label_count}."
            )
        return model

    def _freeze_model(self, model):
        model.eval()
        model.config.residual_in_fp32 = True
        model.model_dtype = INFERENCE_DTYPE
        for module in model.modules():
            if hasattr(module, "residual_in_fp32"):
                module.residual_in_fp32 = True
        # Tied weights share one tensor object, so .to() only moves shared
        # parameters once and is a near-noop for tensors already on target.
        model = model.to(device=self.model_device, dtype=INFERENCE_DTYPE)
        for param in model.parameters():
            param.requires_grad = False
        return model

    def _model_device(self):
        if self.sequence_model is None:
            return self.model_device
        try:
            return next(self.sequence_model.parameters()).device
        except StopIteration:
            return self.model_device

    def _model_dtype(self):
        if self.sequence_model is None:
            return MODEL_INFERENCE_DTYPE
        try:
            return next(self.sequence_model.parameters()).dtype
        except StopIteration:
            return MODEL_INFERENCE_DTYPE

    def _hmm_dtype(self):
        return HMM_INFERENCE_DTYPE

    def _disabled_autocast(self, device):
        try:
            return torch.autocast(device_type=device.type, enabled=False)
        except (AttributeError, RuntimeError, TypeError):
            if device.type == "cuda":
                return torch.cuda.amp.autocast(enabled=False)
            return nullcontext()

    def _species_name_candidates(self):
        raw_name = (self.species_name or "").strip()
        candidates = []
        for candidate in (
            raw_name,
            raw_name.replace(" ", "_"),
            raw_name.replace("_", " "),
        ):
            if candidate and candidate not in candidates:
                candidates.append(candidate)
        return candidates

    def _load_species_embedding_vector(self):
        if self.species_embedding_vector is not None:
            return self.species_embedding_vector
        if not self.species_name:
            return None
        if not os.path.exists(DEFAULT_SPECIES_EMBEDDING_PATH):
            raise FileNotFoundError(
                f"Species embedding file does not exist: {DEFAULT_SPECIES_EMBEDDING_PATH}"
            )

        with open(DEFAULT_SPECIES_EMBEDDING_PATH, "rb") as file_obj:
            species_embeddings = pickle.load(file_obj)
        if not isinstance(species_embeddings, dict):
            raise TypeError(
                "Species embedding file must contain a dict mapping species names to vectors."
            )

        species_embedding = None
        for candidate in self._species_name_candidates():
            if candidate in species_embeddings:
                self.species_embedding_key = candidate
                species_embedding = species_embeddings[candidate]
                break

        if species_embedding is None:
            normalized_keys = {
                str(key).replace(" ", "_"): key for key in species_embeddings.keys()
            }
            for candidate in self._species_name_candidates():
                normalized_candidate = candidate.replace(" ", "_")
                if normalized_candidate in normalized_keys:
                    self.species_embedding_key = normalized_keys[normalized_candidate]
                    species_embedding = species_embeddings[self.species_embedding_key]
                    break

        if species_embedding is None:
            examples = ", ".join(list(map(str, species_embeddings.keys()))[:5])
            raise KeyError(
                f"Species {self.species_name!r} was not found in "
                f"{DEFAULT_SPECIES_EMBEDDING_PATH}. Example keys: {examples}"
            )

        vector = np.asarray(species_embedding, dtype=np.float32).squeeze()
        if vector.ndim != 1:
            raise ValueError(
                f"Species embedding for {self.species_embedding_key!r} must be 1D, "
                f"got shape {vector.shape}."
            )
        self.species_embedding_vector = torch.from_numpy(vector.copy())
        print(
            "Species embedding loaded: "
            f"{self.species_embedding_key} shape={tuple(self.species_embedding_vector.shape)} "
            f"from {DEFAULT_SPECIES_EMBEDDING_PATH}"
        )
        return self.species_embedding_vector

    def _build_species_embedding_batch(self, batch_size):
        species_embedding = self._load_species_embedding_vector()
        if species_embedding is None:
            return None
        return (
            species_embedding.to(device=self.model_device, dtype=self._model_dtype())
            .unsqueeze(0)
            .expand(batch_size, -1)
            .contiguous()
        )

    def load_model(self, summary=True):
        """Load the OrionGeno checkpoint and optional 20-label HMM layer.

        Args:
            summary (bool, optional): If True, print the loaded model structure.
        """
        if not self.model_path:
            raise ValueError("model_path is required and must point to a 20-label OrionGeno checkpoint.")
        if self.model_path:
            self.sequence_model = self._freeze_model(self._load_sequence_model())
            self.model_device = self._model_device()
            print(
                f"Model loaded on {self.model_device}; "
                f"gene labels={self.gene_label_count}; repeat labels={self.repeat_label_count}"
            )
            if summary:
                print(self.sequence_model)
            if self.hmm:
                from .hmm_layer import GenePredHMMLayer

                self.gene_pred_hmm_layer = GenePredHMMLayer(
                    label_dim=self.hmm_label_count,
                    parallel_factor=self.parallel_factor
                )
                self.gene_pred_hmm_layer = self.gene_pred_hmm_layer.to(
                    device=self.model_device,
                    dtype=self._hmm_dtype(),
                )
                self.gene_pred_hmm_layer.cell.recurrent_init()
                self.gene_pred_hmm_layer.reverse_cell.recurrent_init()
                print("20-label HMM layer loaded")
            if self.species_name:
                self._load_species_embedding_vector()

        if summary:
            pass

    def adapt_batch_size(self, adapted_chunksize):
        """Update the forward-pass batch size for the adapted model window.

        The model forward accepts any batch size at runtime, so changing it only
        updates ``self.adapted_batch_size`` and never reloads the checkpoint.
        """
        if adapted_chunksize < self.seq_len:
            self.adapted_batch_size = self.batch_size * self.seq_len // adapted_chunksize
            self.adapted_batch_size = 2 ** int(np.log2(max(1, self.adapted_batch_size)))
        else:
            self.adapted_batch_size = self.batch_size

    def init_fasta(self, genome_path=None, chunk_len=None, min_seq_len=0):
        if genome_path is None:
            genome_path = self.genome_path
        if chunk_len is None:
            chunk_len = self.seq_len
        if self.genome:
            fasta = GenomeSequences(
                genome=self.genome,
                chunksize=chunk_len,
                overlap=0,
                min_seq_len=min_seq_len,
                upper_only=self.upper_only,
            )
        else:
            fasta = GenomeSequences(
                fasta_file=genome_path,
                chunksize=chunk_len,
                overlap=0,
                min_seq_len=min_seq_len,
                upper_only=self.upper_only,
            )
        return fasta

    def load_genome_data(
        self,
        fasta_object,
        seq_names,
        strand="",
        upper_only=True,
        flank_size=0,
        encode=True,
    ):
        if strand is None:
            strand = self.strand
        if encode:
            fasta_object.encode_sequences(seq=seq_names)

        f_chunk, coords, model_chunksize = fasta_object.get_flat_chunks(
            strand=strand,
            coords=True,
            sequence_names=seq_names,
            adapt_chunksize=True,
            parallel_factor=self.parallel_factor,
            flank_size=flank_size,
        )
        if not upper_only:
            f_chunk = f_chunk[:, :, :9]
        return f_chunk, coords, model_chunksize

    def _call_sequence_model(self, input_ids):
        species_embedding = self._build_species_embedding_batch(input_ids.shape[0])
        if species_embedding is None:
            return self.sequence_model(input_ids)
        return self.sequence_model(input_ids, species_embedding=species_embedding)

    def _require_20_label_gene_head(self, gene_head):
        if not torch.is_tensor(gene_head):
            raise TypeError(f"Gene head must be a tensor, got {type(gene_head)!r}.")
        if gene_head.shape[-1] != GENE_LABEL_COUNT:
            raise ValueError(
                f"Gene head must output exactly {GENE_LABEL_COUNT} labels, got {gene_head.shape[-1]}."
            )
        return gene_head

    def _extract_gene_head(self, model_output):
        if torch.is_tensor(model_output):
            return self._require_20_label_gene_head(model_output)
        gene_logits = getattr(model_output, "gene_logits", None)
        if gene_logits is not None:
            return self._require_20_label_gene_head(gene_logits)
        logits = getattr(model_output, "logits", None)
        if logits is not None:
            if isinstance(logits, (tuple, list)):
                return self._require_20_label_gene_head(logits[0])
            return self._require_20_label_gene_head(logits)
        if isinstance(model_output, (tuple, list)):
            for item in model_output:
                if torch.is_tensor(item) and item.dim() >= 3:
                    return self._require_20_label_gene_head(item)
                if isinstance(item, (tuple, list)) and item and torch.is_tensor(item[0]):
                    return self._require_20_label_gene_head(item[0])
        raise TypeError(f"Could not extract gene logits from model output type {type(model_output)!r}.")

    def _require_2_label_repeat_head(self, repeat_head):
        if not torch.is_tensor(repeat_head):
            raise TypeError(f"Repeat head must be a tensor, got {type(repeat_head)!r}.")
        if repeat_head.shape[-1] != REPEAT_LABEL_COUNT:
            raise ValueError(
                f"Repeat head must output exactly {REPEAT_LABEL_COUNT} labels, got {repeat_head.shape[-1]}."
            )
        return repeat_head

    def _extract_repeat_head(self, model_output):
        repeat_logits = getattr(model_output, "repeat_logits", None)
        if repeat_logits is not None:
            return self._require_2_label_repeat_head(repeat_logits)
        logits = getattr(model_output, "logits", None)
        if logits is not None:
            if isinstance(logits, (tuple, list)) and len(logits) > 1:
                return self._require_2_label_repeat_head(logits[1])
            if torch.is_tensor(logits) and logits.shape[-1] == REPEAT_LABEL_COUNT:
                return self._require_2_label_repeat_head(logits)
        if isinstance(model_output, (tuple, list)):
            for item in model_output:
                if torch.is_tensor(item) and item.dim() >= 3 and item.shape[-1] == REPEAT_LABEL_COUNT:
                    return self._require_2_label_repeat_head(item)
                if isinstance(item, (tuple, list)) and len(item) > 1 and torch.is_tensor(item[1]):
                    return self._require_2_label_repeat_head(item[1])
        raise TypeError(f"Could not extract repeat logits from model output type {type(model_output)!r}.")

    def _as_probabilities(self, values, dtype=None):
        if dtype is None:
            dtype = values.dtype if values.dtype.is_floating_point else self._model_dtype()
        values = values.to(dtype=dtype)
        sums = values.sum(dim=-1)
        looks_like_probabilities = (
            torch.isfinite(values).all()
            and values.min() >= -1e-6
            and torch.mean(torch.abs(sums - 1.0)) < 1e-3
        )
        if looks_like_probabilities:
            probabilities = values.clamp_min(0.0) / sums.clamp_min(1e-12).unsqueeze(-1)
            return probabilities.to(dtype=dtype)
        return torch.softmax(values, dim=-1, dtype=dtype)

    def _release_cuda_cache(self, force=False):
        if self.cuda_cache_clear_interval <= 0 and not force:
            return
        self._cuda_cache_release_calls += 1
        if (
            not force
            and self._cuda_cache_release_calls % self.cuda_cache_clear_interval != 0
        ):
            return
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except RuntimeError:
                pass

    def _prepare_gene_hmm_inputs(self, gene_outputs, device):
        dtype = self._hmm_dtype()
        if torch.is_tensor(gene_outputs):
            gene_outputs = gene_outputs.to(device=device, dtype=dtype)
        else:
            gene_outputs = torch.as_tensor(gene_outputs, dtype=dtype, device=device)

        probabilities = self._as_probabilities(gene_outputs, dtype=dtype)
        num_classes = probabilities.shape[-1]
        if num_classes != GENE_LABEL_COUNT:
            raise ValueError(
                f"Gene head must output exactly {GENE_LABEL_COUNT} labels, got {num_classes}."
            )
        return probabilities

    def sequence_prediction(
        self,
        inp_chunks,
        save=True,
        batch_size=None,
        output_gene=True,
        output_repeat=False,
    ):
        """Generate raw model-head predictions from OrionGeno.

        Arguments:
            inp_chunks (np.ndarray): One-hot encoded nucleotide chunks for the sequence model
                (converted to token ids internally).
            save (bool): A flag to indicate whether the predictions should be saved/loaded to/from a file.
            output_gene (bool): Return gene-head probabilities for downstream HMM/annotation generation.
            output_repeat (bool): Return repeat-head binary argmax labels without smoothing.

        Returns:
            torch.Tensor or dict: Gene-head probability tensor for the legacy gene-only path, otherwise a
            dictionary containing requested heads. Repeat output is already argmax-decoded.
        """
        if not output_gene and not output_repeat:
            raise ValueError("At least one of output_gene or output_repeat must be True.")
        if not batch_size:
            batch_size = self.adapted_batch_size
        batch_size = max(1, int(batch_size))
        gene_predictions = []
        repeat_predictions = []
        legacy_gene_only = output_gene and not output_repeat

        print(f"### Model input chunks shape: {inp_chunks.shape}")
        if (
            save
            and self.temp_dir
            and legacy_gene_only
            and os.path.exists(f"{self.temp_dir}/sequence_predictions.npz")
        ):
            sequence_predictions = np.load(f"{self.temp_dir}/sequence_predictions.npz")
            sequence_predictions = sequence_predictions["array1"]
            return sequence_predictions

        heads_cache = f"{self.temp_dir}/sequence_predictions_heads.npz" if self.temp_dir else ""
        if save and self.temp_dir and not legacy_gene_only and os.path.exists(heads_cache):
            cached = np.load(heads_cache)
            result = {}
            if output_gene:
                result["gene"] = cached["gene"]
            if output_repeat:
                result["repeat"] = cached["repeat"]
            return result

        chunk_count = inp_chunks.shape[0]
        chunk_pos = 0
        with tqdm(total=chunk_count, desc="Model prediction:") as progress:
            while chunk_pos < chunk_count:
                current_batch = min(batch_size, chunk_count - chunk_pos)
                input_ids = None
                model_output = None
                gene_head = None
                repeat_head = None
                probabilities = None
                try:
                    start_pos = chunk_pos
                    end_pos = chunk_pos + current_batch
                    input_ids = torch.as_tensor(inp_chunks[start_pos:end_pos])
                    input_ids = torch.argmax(input_ids, dim=-1).to(self.model_device)
                    with torch.inference_mode():
                        model_output = self._call_sequence_model(input_ids)
                        y_gene = None
                        y_repeat = None
                        if output_gene:
                            gene_head = self._extract_gene_head(model_output)
                            probability_dtype = self._hmm_dtype() if self.hmm else None
                            probabilities = self._as_probabilities(
                                gene_head,
                                dtype=probability_dtype,
                            )
                            y_gene = probabilities.detach().to(dtype=torch.float16).cpu()
                        if output_repeat:
                            repeat_head = self._extract_repeat_head(model_output)
                            y_repeat = repeat_head.argmax(dim=-1).detach().cpu().numpy()
                            if len(y_repeat.shape) == 1:
                                y_repeat = np.expand_dims(y_repeat, 0)
                            y_repeat = y_repeat.astype(np.int8, copy=False)
                    if output_gene:
                        gene_predictions.append(y_gene)
                    if output_repeat:
                        repeat_predictions.append(y_repeat)
                    chunk_pos = end_pos
                    progress.update(current_batch)
                    del input_ids, model_output, gene_head, repeat_head, probabilities
                    self._release_cuda_cache()
                except torch.cuda.OutOfMemoryError:
                    del input_ids, model_output, gene_head, repeat_head, probabilities
                    self._release_cuda_cache(force=True)
                    if current_batch <= 1:
                        raise
                    next_batch = max(1, current_batch // 2)
                    logging.warning(
                        "Model-forward batch of %d hit CUDA OOM; retrying with batch_size=%d.",
                        current_batch,
                        next_batch,
                    )
                    batch_size = next_batch
                    self.adapted_batch_size = min(self.adapted_batch_size, next_batch)
        result = {}
        if output_gene:
            result["gene"] = torch.cat(gene_predictions, dim=0)
        if output_repeat:
            result["repeat"] = np.concatenate(repeat_predictions, axis=0)

        if save and self.temp_dir and legacy_gene_only:
            sequence_predictions = result["gene"]
            np.savez(
                f"{self.temp_dir}/sequence_predictions.npz",
                array1=sequence_predictions.float().numpy(),
            )
            return sequence_predictions
        if save and self.temp_dir and not legacy_gene_only:
            cache_result = dict(result)
            if output_gene:
                cache_result["gene"] = result["gene"].float().numpy()
            np.savez(heads_cache, **cache_result)
        if legacy_gene_only:
            return result["gene"]
        return result

    def hmm_predictions_filtered(
        self, inp_chunks, sequence_predictions, save=True, batch_size=None
    ):
        """Generate HMM predictions after a lightweight pre-filter.
        It first analyzes the gene label probabilities from OrionGeno over
        windows of 200 base pairs (bp) in length. The HMM makes predictions on
        an example only if there's at least one window where the average summed
        probability across the CDS-mapped states is 0.8 or higher. If no such window exists,
        the HMM will skip making predictions for that example, and all positions
        within it are labeled as intergenic region.

        Arguments:
            inp_chunks (np.array): One hot encoded representation of the input nucleotide sequence.
            sequence_predictions (np.array): Gene label probabilities from OrionGeno.
            save (bool): A flag to indicate whether the predictions should be saved/loaded to/from a file.

        Returns:
            np.ndarray: HMM-decoded labels.
        """
        if not batch_size:
            batch_size = self.adapted_batch_size

        # Decode batch is decoupled from the model-forward batch: when set, use
        # it to fill larger Viterbi batches and amortize kernel-launch overhead.
        decode_batch = self.hmm_decode_batch if self.hmm_decode_batch > 0 else batch_size

        print("### Running HMM Viterbi filtered decoding")

        hmm_predictions = []

        if (
            save
            and self.temp_dir
            and os.path.exists(f"{self.temp_dir}/hmm_predictions.npy")
        ):
            hmm_predictions = np.load(f"{self.temp_dir}/hmm_predictions.npy")
            return hmm_predictions

        prefilter_probs = (
            sequence_predictions.detach().cpu().numpy()
            if torch.is_tensor(sequence_predictions)
            else np.asarray(sequence_predictions)
        )
        if prefilter_probs.ndim != 3 or prefilter_probs.shape[-1] != GENE_LABEL_COUNT:
            raise ValueError(
                "HMM filtered prediction expects probabilities with shape "
                f"(num_chunks, chunk_length, {GENE_LABEL_COUNT}), got {prefilter_probs.shape}."
            )

        batch_i = []
        hmm_predictions = np.zeros(
            (inp_chunks.shape[0], inp_chunks.shape[1]), dtype=np.int16
        )
        for i in tqdm(
            range(inp_chunks.shape[0]),
            desc=f"HMM filtered prediction: {hmm_predictions.shape}",
        ):
            if self._chunk_has_cds_signal(prefilter_probs[i]):
                batch_i += [i]
            else:
                hmm_predictions[i] = 0

            if batch_i and (
                len(batch_i) == decode_batch
                or i == inp_chunks.shape[0] - 1
            ):
                y_hmm = self._decode_vit_batch(
                    inp_chunks, sequence_predictions, batch_i
                )
                if len(y_hmm.shape) == 1:
                    y_hmm = np.expand_dims(y_hmm, 0)
                for j1, j2 in enumerate(batch_i):
                    hmm_predictions[j2] = y_hmm[j1]
                batch_i = []
                del y_hmm
                self._release_cuda_cache()

        if save and self.temp_dir:
            np.save(f"{self.temp_dir}/hmm_predictions.npy", hmm_predictions)
        return hmm_predictions

    def _decode_vit_batch(self, inp_chunks, sequence_predictions, batch_i):
        """Run Viterbi on a batch of chunk indices, halving on CUDA OOM.

        Larger HMM decode batches give the CPU decoder more batch-level work and
        amortize per-step kernel-launch overhead on the GPU fallback path. The
        emission step's memory grows with the batch. If the device runs out of
        memory we recursively split the batch in half and retry, so a too large
        ORIONGENO_HMM_DECODE_BATCH degrades gracefully instead of crashing.
        """
        try:
            y_hmm = (
                self.predict_vit(
                    inp_chunks[batch_i], sequence_predictions[batch_i]
                )
                .cpu()
                .numpy()
                .squeeze()
            )
            if len(y_hmm.shape) == 1:
                y_hmm = np.expand_dims(y_hmm, 0)
            return y_hmm
        except torch.cuda.OutOfMemoryError:
            if len(batch_i) <= 1:
                raise
            self._release_cuda_cache(force=True)
            mid = len(batch_i) // 2
            logging.warning(
                "HMM decode batch of %d hit CUDA OOM; retrying as %d + %d.",
                len(batch_i),
                mid,
                len(batch_i) - mid,
            )
            left = self._decode_vit_batch(
                inp_chunks, sequence_predictions, batch_i[:mid]
            )
            right = self._decode_vit_batch(
                inp_chunks, sequence_predictions, batch_i[mid:]
            )
            return np.concatenate([left, right], axis=0)

    def get_predictions(
        self,
        inp_chunks,
        save=True,
        encoding_layer_oracle=None,
        batch_size=None,
    ):
        """Run the sequence model and optional HMM for input chunks.

        Args:
            inp_chunks (np.ndarray): Input chunks for which to get predictions.
            save (bool): A flag to indicate whether the predictions should be saved/loaded to/from a file.
            encoding_layer_oracle (np.ndarray or None): Pre-computed gene-label predictions. When provided
                (not None), the sequence model forward pass is skipped and these predictions are used directly.
                Use for debugging.

        Returns:
            np.ndarray: HMM predictions for all chunks.
        """
        if not batch_size:
            batch_size = self.adapted_batch_size

        start_time = time.time()
        if encoding_layer_oracle is not None:
            encoding_layer_pred = encoding_layer_oracle
        else:
            encoding_layer_pred = self.sequence_prediction(
                inp_chunks, save=save, batch_size=batch_size
            )

        self.sequence_predictions = encoding_layer_pred
        model_end = time.time()
        duration = model_end - start_time
        print(f"OrionGeno sequence model took {duration/60:.4f} minutes to execute.")
        if not self.hmm:
            if torch.is_tensor(encoding_layer_pred):
                encoding_layer_pred = encoding_layer_pred.argmax(dim=-1).cpu().numpy()
            else:
                encoding_layer_pred = np.argmax(encoding_layer_pred, axis=-1)
            return encoding_layer_pred

        hmm_predictions = self.hmm_predictions_filtered(
            inp_chunks, encoding_layer_pred, save=save, batch_size=batch_size
        )
        hmm_end = time.time()
        duration = hmm_end - model_end
        print(f"HMM took {duration/60:.4f} minutes to execute.")
        return hmm_predictions

    def get_prediction_outputs(
        self,
        inp_chunks,
        save=True,
        encoding_layer_oracle=None,
        batch_size=None,
        output_gene=True,
        output_repeat=False,
    ):
        """Run requested output heads, applying HMM only to gene predictions."""
        if not output_gene and not output_repeat:
            return {}
        if not batch_size:
            batch_size = self.adapted_batch_size

        start_time = time.time()
        if encoding_layer_oracle is not None:
            if output_repeat:
                raise NotImplementedError("encoding_layer_oracle only supplies gene predictions.")
            raw_heads = {"gene": encoding_layer_oracle}
        else:
            raw_heads = self.sequence_prediction(
                inp_chunks,
                save=save,
                batch_size=batch_size,
                output_gene=output_gene,
                output_repeat=output_repeat,
            )
            if output_gene and not output_repeat:
                raw_heads = {"gene": raw_heads}

        outputs = {}
        model_end = time.time()
        duration = model_end - start_time
        print(f"OrionGeno sequence model took {duration/60:.4f} minutes to execute.")

        if output_gene:
            gene_predictions = raw_heads["gene"]
            self.sequence_predictions = gene_predictions
            if not self.hmm:
                if torch.is_tensor(gene_predictions):
                    outputs["gene"] = gene_predictions.argmax(dim=-1).cpu().numpy()
                else:
                    outputs["gene"] = np.argmax(gene_predictions, axis=-1)
            else:
                outputs["gene"] = self.hmm_predictions_filtered(
                    inp_chunks,
                    gene_predictions,
                    save=save,
                    batch_size=batch_size,
                )
            hmm_end = time.time()
            duration = hmm_end - model_end
            print(f"HMM took {duration/60:.4f} minutes to execute.")

        if output_repeat:
            outputs["repeat"] = raw_heads["repeat"]

        return outputs
    def predict_vit(self, x, gene_outputs):
        """Perform prediction using Viterbi decoding on gene label probabilities.

        This method applies the Viterbi algorithm to OrionGeno gene label probabilities
        to find the most likely sequence of hidden states. When enabled, candidate genic
        regions are detected first to reduce Viterbi workload on intergenic sequences.

        Args:
            x (np.ndarray): Encoded nucleotide input for the current batch.
            gene_outputs (np.ndarray): Gene label probabilities used for Viterbi decoding.

        Returns:
            torch.Tensor: Predicted state sequence after Viterbi decoding.
        """
        device = self._model_device()
        gene_probabilities = self._prepare_gene_hmm_inputs(gene_outputs, device)

        nucleotides_ids = torch.as_tensor(x, dtype=self._hmm_dtype(), device=device)
        if not self.upper_only and nucleotides_ids.shape[-1] >= 9:
            nuc = torch.stack(
                [
                    nucleotides_ids[..., 0] + nucleotides_ids[..., 5],
                    nucleotides_ids[..., 1] + nucleotides_ids[..., 6],
                    nucleotides_ids[..., 2] + nucleotides_ids[..., 7],
                    nucleotides_ids[..., 3] + nucleotides_ids[..., 8],
                    nucleotides_ids[..., 4],
                ],
                dim=-1,
            )
        elif nucleotides_ids.shape[-1] >= 5:
            nuc = nucleotides_ids[..., :5]
        else:
            raise ValueError(f"Expected nucleotide input with at least 5 channels, got {nucleotides_ids.shape}.")

        if not self.hmm:
            return gene_probabilities.argmax(dim=-1)

        # Candidate genic region filtering (controlled by ORIONGENO_GENIC_REGION_FILTER)
        # Default OFF after validation showed it slows down gene-dense sequences.
        # Benefits expected on sparse genomes (>60% intergenic) with long scaffolds (>10Mb).
        # Set ORIONGENO_GENIC_REGION_FILTER=1 to force enable, or =auto for heuristic.
        filter_mode = os.environ.get("ORIONGENO_GENIC_REGION_FILTER", "0").lower()

        if filter_mode == "auto":
            # Heuristic: enable only for long sequences (likely sparse genomes)
            batch_size, seq_len, num_states = gene_probabilities.shape
            use_region_filter = seq_len > 5_000_000
        else:
            use_region_filter = filter_mode not in {"0", "false", "no", "off"}

        with self._disabled_autocast(device):
            if not use_region_filter:
                # Original path: decode the full sequence
                return self.gene_pred_hmm_layer.viterbi(gene_probabilities, nucleotides=nuc)

            # Import here to avoid overhead when disabled
            from .hmm_layer.genic_region_filter import filter_genic_regions

            batch_size, seq_len, num_states = gene_probabilities.shape
            if batch_size != 1:
                # Region filtering only supports batch=1; fall back to full decode
                return self.gene_pred_hmm_layer.viterbi(gene_probabilities, nucleotides=nuc)

            # Detect candidate genic regions
            genic_regions = filter_genic_regions(
                gene_probabilities,
                min_avg_threshold=0.1,
                max_threshold=0.5,
                min_cds_length=50,
                buffer_size=100,
                step=50,
            )

            # If the entire sequence is one region, decode as normal
            if len(genic_regions) == 1 and genic_regions[0] == (0, seq_len):
                return self.gene_pred_hmm_layer.viterbi(gene_probabilities, nucleotides=nuc)

            # Decode each genic region independently and fill intergenic with state 0
            viterbi_paths = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

            for start, end in genic_regions:
                # Slice the region (end is inclusive in our representation)
                region_probs = gene_probabilities[:, start:end+1, :]
                region_nuc = nuc[:, start:end+1, :]

                # Decode this region
                region_path = self.gene_pred_hmm_layer.viterbi(region_probs, nucleotides=region_nuc)

                # Copy into full path
                viterbi_paths[:, start:end+1] = region_path

            return viterbi_paths


    def get_tx_from_range(self, range_):
        """Split feature ranges into complete and boundary-fragmented transcripts.

        Parameters:
        - range_ (list): Feature ranges whose first item is the feature type.

        Returns:
        - initial_tx (list): Feature ranges for the first boundary-fragmented transcript.
        - txs (list): Complete transcripts found within the range list.
        - current_tx (list): Feature ranges for the last boundary-fragmented transcript.
        """

        txs = []
        current_tx = []
        initial_tx = []

        for region in range_:
            if region[0] == "intergenic":
                if current_tx:
                    txs.append(current_tx)
                    current_tx = []
            else:
                current_tx.append(region)
        if range_[0][0] != "intergenic" and txs:
            initial_tx = txs[0]
            txs = txs[1:]
        return initial_tx, txs, current_tx

    def create_gene_annotation(
        self,
        y_label,
        coords,
        f_chunks,
        out_file="",
        strand="+",
        correct_y_label=None,
        anno=None,
        tx_id=0,
        filt=True,
        allow_boundary_reprediction=True,
    ):
        """Create gene annotation records from predictions.

        This method translates model predictions into annotation records. Optional boundary
        re-prediction reruns a centered window around disagreeing chunk borders.

        Args:
            y_label (np.ndarray): The array of encoded labels predicted by the model.
            coords (np.ndarray): The array of genomic coordinates: [seq_name, strand, chunk_start, chunk_end].
            out_file (str): Path to the output annotation file.
            f_chunks (np.array): One hot encoded nucleotide sequence.
            correct_y_label (np.array): Correct y_label for debugging.
        """
        batch_size = max(1, self.adapted_batch_size)

        # Convert minus-strand batches back to genomic coordinate order.
        if strand == "-":
            y_label = y_label[::-1, ::-1]
            f_chunks = f_chunks[::-1]
            coords = coords[::-1]
            if correct_y_label is not None:
                correct_y_label = correct_y_label[::-1, ::-1]

        # Accumulate transcript feature ranges by sequence name.
        ranges = {}
        if not anno:
            anno = Anno(out_file, "anno")

        re_pred_inp = []
        re_correct_y_label = []
        re_pred_meta = []

        if allow_boundary_reprediction:
            for i in range(y_label.shape[0] - 1):
                if coords[i][0] == coords[i + 1][0] and not (
                    y_label[i, -1] == y_label[i + 1, 0]
                ):
                    left_context = int(f_chunks[i].shape[0])
                    right_context = int(f_chunks[i + 1].shape[0])
                    if left_context <= 0 or right_context <= 0:
                        continue
                    if left_context != right_context:
                        raise ValueError(
                            "Boundary re-prediction requires equal adjacent chunk "
                            f"lengths, got {left_context} and {right_context}."
                        )

                    if coords[i][1] == "+":
                        re_pred_inp.append(
                            np.concatenate(
                                [
                                    f_chunks[i],
                                    f_chunks[i + 1],
                                ],
                                axis=0,
                            )
                        )
                        if correct_y_label is not None:
                            re_correct_y_label.append(
                                np.concatenate(
                                    [
                                        correct_y_label[i],
                                        correct_y_label[i + 1],
                                    ],
                                    axis=0,
                                )
                            )
                        left_replace = left_context
                        right_replace = right_context
                    else:
                        re_pred_inp.append(
                            np.concatenate(
                                [
                                    f_chunks[i + 1],
                                    f_chunks[i],
                                ],
                                axis=0,
                            )
                        )
                        if correct_y_label is not None:
                            re_correct_y_label.append(
                                np.concatenate(
                                    [
                                        correct_y_label[i + 1],
                                        correct_y_label[i],
                                    ],
                                    axis=0,
                                )
                            )
                        left_replace = left_context
                        right_replace = right_context

                    re_pred_meta.append(
                        {
                            "index": i,
                            "left_replace": left_replace,
                            "right_replace": right_replace,
                            "reverse": coords[i][1] == "-",
                        }
                    )

        # Re-predict ambiguous chunk boundaries using both complete adjacent
        # chunks, producing a centered window twice the normal chunk length.
        if re_pred_inp:
            re_pred_inp = np.stack(re_pred_inp, axis=0)
            re_correct_y_label = (
                np.stack(re_correct_y_label, axis=0) if re_correct_y_label else None
            )
            boundary_batch_size = max(
                1,
                batch_size // BOUNDARY_REPREDICTION_LENGTH_FACTOR,
            )
            re_pred = self.get_predictions(
                re_pred_inp,
                save=False,
                batch_size=boundary_batch_size,
                encoding_layer_oracle=re_correct_y_label,
            )

            for current_re, meta in zip(re_pred, re_pred_meta):
                if meta["reverse"]:
                    current_re = current_re[::-1]
                i = meta["index"]
                left_replace = meta["left_replace"]
                right_replace = meta["right_replace"]
                y_label[i, -left_replace:] = current_re[:left_replace]
                y_label[i + 1, :right_replace] = current_re[
                    left_replace : left_replace + right_replace
                ]

        end_fragment = []
        current_seq_name = None

        for i, (y, c) in enumerate(zip(y_label, coords)):
            if c[0] != current_seq_name:
                current_seq_name = c[0]
                end_fragment = []

            valid_label_len = max(0, int(c[3]) - int(c[2]) + 1)
            if valid_label_len <= 0:
                continue
            if valid_label_len < len(y):
                y = y[:valid_label_len]
            y_ranges = self.get_ranges(y, c[2])
            is_ir = "intergenic" in [r[0] for r in y_ranges]
            start_fragment, txs, new_end_fragment = self.get_tx_from_range(y_ranges)

            # Start a new sequence bucket on the first chunk for each scaffold.
            if c[0] not in ranges:
                ranges[c[0]] = []

            # Join transcript fragments that continue across adjacent chunks.
            if (
                is_ir
                and end_fragment
                and start_fragment
                and i > 0
                and y_label[i - 1, -1] == y_label[i, 0]
            ):
                end_fragment[-1][2] = start_fragment[0][2]
                end_fragment += start_fragment[1:]
                ranges[c[0]] += [end_fragment]
            if is_ir and txs:
                ranges[c[0]] += txs
            if is_ir:
                end_fragment = new_end_fragment

        for seq in ranges:
            phase = -1
            for tx in ranges[seq]:
                tx_id += 1
                t_id = f"g{tx_id}.t1"
                g_id = f"g{tx_id}"
                phase = 0
                anno.transcript_update(t_id, g_id, seq, strand)
                anno.genes_update(g_id, t_id)
                for r in tx:
                    line = [
                        seq,
                        "OrionGeno",
                        r[0],
                        r[1],
                        r[2],
                        ".",
                        strand,
                        phase,
                        f'gene_id "{g_id}"; transcript_id "{t_id}";',
                    ]
                    anno.transcripts[t_id].add_line(line)
                    if r[0] == "CDS":
                        phase = (3 - (r[2] - r[1] + 1 - phase) % 3) % 3

        remove_tx = []
        for tx in anno.transcripts.values():
            tx.check_splits()
            if filt and tx.get_cds_len() < 201:
                remove_tx.append(tx.id)
            else:
                tx.redo_phase()

        for tx in remove_tx:
            anno.transcripts.pop(tx)

        if out_file:
            anno.norm_tx_format()
            anno.find_genes()
            anno.write_anno(out_file)

        return anno, tx_id

    def get_ranges(self, encoded_labels, offset=0):
        """Convert encoded label runs into genomic feature ranges.

        This method reduces the 20-label state sequence to gene feature classes,
        then groups consecutive positions with the same feature label.

        Args:
            encoded_labels (Iterable[int]): Encoded labels representing genomic features.

        Returns:
            List[List]: A list of three-element lists where each list contains the feature type
            as a string and the start and end points as integers.
        """
        arr = np.array(encoded_labels)

        arr = self.reduce_label(arr)

        # Split the label vector at every class change.
        change_points = np.where(np.diff(arr) != 0)[0]
        start_points = np.insert(change_points + 1, 0, 0)
        end_points = np.append(change_points, arr.size - 1)

        features = GENE_FEATURE_NAMES
        ranges = [
            [features[arr[start]], start + offset, end + offset]
            for start, end in zip(start_points, end_points)
        ]

        return ranges
