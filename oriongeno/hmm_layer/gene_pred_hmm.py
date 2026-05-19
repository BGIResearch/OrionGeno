import json

import torch

from .BaseRNN import BaseRNN
from .Bidirectional import Bidirectional
from .Initializers import make_20_class_emission_kernel
from .MsaHMMLayer import MsaHmmLayer, _state_posterior_log_probs_impl
from .MsaHmmCell import HmmCell
from .TotalProbabilityCell import TotalProbabilityCell
from .gene_pred_hmm_emitter import GenePredHMMEmitter
from .gene_pred_hmm_topology import GENE_PRED_BASE_LABEL_DIM, expanded_state_names
from .gene_pred_hmm_transitioner import GenePredMultiHMMTransitioner

try:
    from line_profiler import profile
except ImportError:
    def profile(func):
        return func


def load_config_from_json(config_file):
    with open(config_file, "r", encoding="utf-8") as file_obj:
        config = json.load(file_obj)

    params = {}
    for key in ("num_models", "num_copies", "share_intron_parameters", "starting_distribution_init", "label_dim", "emission_smoothing"):
        if key in config:
            params[key] = config[key]

    if "initial_lengths" in config:
        lengths = config["initial_lengths"]
        if "exon" in lengths:
            params["initial_exon_len"] = lengths["exon"]
        if "intron" in lengths:
            params["initial_intron_len"] = lengths["intron"]
        if "intergenic" in lengths:
            params["initial_ir_len"] = lengths["intergenic"]
        if "utr" in lengths:
            params["initial_utr_len"] = lengths["utr"]
        elif "five_utr" in lengths:
            params["initial_utr_len"] = lengths["five_utr"]
        elif "three_utr" in lengths:
            params["initial_utr_len"] = lengths["three_utr"]
        if "utr_intron" in lengths:
            params["initial_utr_intron_len"] = lengths["utr_intron"]

    for key in ("start_codons", "stop_codons", "intron_begin_pattern", "intron_end_pattern"):
        if key in config:
            params[key] = [tuple(item) for item in config[key]]

    trainable = config.get("trainable", {})
    mapping = {
        "emissions": "trainable_emissions",
        "transitions": "trainable_transitions",
        "starting_distribution": "trainable_starting_distribution",
        "nucleotides_at_exons": "trainable_nucleotides_at_exons",
    }
    for key, mapped_key in mapping.items():
        if key in trainable:
            params[mapped_key] = trainable[key]

    return params


class GenePredHMMLayer(MsaHmmLayer):
    """PyTorch gene-prediction HMM layer with the default 20-label topology."""

    def __init__(
        self,
        config_path=None,
        num_models=1,
        num_copies=1,
        start_codons=[("ATG", 1.0)],
        stop_codons=[("TAG", 0.34), ("TAA", 0.33), ("TGA", 0.33)],
        intron_begin_pattern=[("NGT", 0.99), ("NGC", 0.005), ("NAT", 0.005)],
        intron_end_pattern=[("AGN", 0.99), ("ACN", 0.01)],
        initial_exon_len=200,
        initial_intron_len=4500,
        initial_ir_len=10000,
        initial_utr_len=60,
        initial_utr_intron_len=None,
        label_dim=GENE_PRED_BASE_LABEL_DIM,
        emission_smoothing=1e-2,
        emitter_init=None,
        starting_distribution_init="zeros",
        trainable_emissions=False,
        trainable_transitions=False,
        trainable_starting_distribution=False,
        trainable_nucleotides_at_exons=False,
        emit_embeddings=False,
        embedding_dim=None,
        full_covariance=False,
        embedding_kernel_init="random_normal",
        initial_variance=0.1,
        temperature=96.0,
        share_intron_parameters=False,
        simple=False,
        variance_l2_lambda=0.01,
        disable_metrics=True,
        parallel_factor=1,
        use_border_hints=False,
        device=None,
        **kwargs,
    ):
        if config_path is not None:
            json_params = load_config_from_json(config_path)
            num_models = json_params.get("num_models", num_models)
            num_copies = json_params.get("num_copies", num_copies)
            share_intron_parameters = json_params.get("share_intron_parameters", share_intron_parameters)
            initial_exon_len = json_params.get("initial_exon_len", initial_exon_len)
            initial_intron_len = json_params.get("initial_intron_len", initial_intron_len)
            initial_ir_len = json_params.get("initial_ir_len", initial_ir_len)
            initial_utr_len = json_params.get("initial_utr_len", initial_utr_len)
            initial_utr_intron_len = json_params.get(
                "initial_utr_intron_len",
                initial_utr_intron_len,
            )
            label_dim = json_params.get("label_dim", label_dim)
            emission_smoothing = float(json_params.get("emission_smoothing", emission_smoothing))
            start_codons = json_params.get("start_codons", start_codons)
            stop_codons = json_params.get("stop_codons", stop_codons)
            intron_begin_pattern = json_params.get("intron_begin_pattern", intron_begin_pattern)
            intron_end_pattern = json_params.get("intron_end_pattern", intron_end_pattern)
            trainable_emissions = json_params.get("trainable_emissions", trainable_emissions)
            trainable_transitions = json_params.get("trainable_transitions", trainable_transitions)
            trainable_starting_distribution = json_params.get("trainable_starting_distribution", trainable_starting_distribution)
            trainable_nucleotides_at_exons = json_params.get("trainable_nucleotides_at_exons", trainable_nucleotides_at_exons)
            starting_distribution_init = json_params.get("starting_distribution_init", starting_distribution_init)

        if emitter_init is None:
            if label_dim != GENE_PRED_BASE_LABEL_DIM:
                raise ValueError(
                    f"The 20-label topology expects label_dim={GENE_PRED_BASE_LABEL_DIM}, got {label_dim}."
                )
            emitter_init = make_20_class_emission_kernel(
                smoothing=emission_smoothing,
                num_copies=num_copies,
                num_models=num_models,
            )

        self.num_models = num_models
        self.num_copies = num_copies
        self.start_codons = start_codons
        self.stop_codons = stop_codons
        self.intron_begin_pattern = intron_begin_pattern
        self.intron_end_pattern = intron_end_pattern
        self.emitter_init = emitter_init
        self.initial_exon_len = initial_exon_len
        self.initial_intron_len = initial_intron_len
        self.initial_ir_len = initial_ir_len
        self.initial_utr_len = initial_utr_len
        self.initial_utr_intron_len = (
            initial_utr_intron_len if initial_utr_intron_len is not None else initial_intron_len
        )
        self.starting_distribution_init = starting_distribution_init
        self.trainable_emissions = trainable_emissions
        self.trainable_transitions = trainable_transitions
        self.trainable_starting_distribution = trainable_starting_distribution
        self.trainable_nucleotides_at_exons = trainable_nucleotides_at_exons
        self.emit_embeddings = emit_embeddings
        self.embedding_dim = embedding_dim
        self.full_covariance = full_covariance
        self.embedding_kernel_init = embedding_kernel_init
        self.initial_variance = initial_variance
        self.temperature = temperature
        self.share_intron_parameters = share_intron_parameters
        self.simple = simple
        self.variance_l2_lambda = variance_l2_lambda
        self.disable_metrics = disable_metrics
        self.use_border_hints = use_border_hints
        self.device = device
        self.dim = int(self.emitter_init.shape[-1])
        self.state_names = expanded_state_names(self.num_copies)
        super(GenePredHMMLayer, self).__init__(parallel_factor=parallel_factor)

    def build(self):
        if hasattr(self, "built") and self.built:
            return

        self.cell, self.reverse_cell = self.create_cell()
        self.rnn = BaseRNN(self.cell, batch_first=True, return_sequences=True, return_state=True)
        self.rnn_backward = BaseRNN(
            self.reverse_cell,
            batch_first=True,
            return_sequences=True,
            return_state=True,
            reverse=self.reverse_cell.reverse,
        )
        self.bidirectional_rnn = Bidirectional(
            self.rnn,
            merge_mode="concat" if self.parallel_factor > 1 else "sum",
            backward_layer=self.rnn_backward,
        )
        self.bidirectional_rnn.forward_layer = self.rnn
        self.bidirectional_rnn.backward_layer = self.rnn_backward

        if self.parallel_factor > 1:
            self.total_prob_cell = TotalProbabilityCell(self.cell)
            self.total_prob_cell_rev = TotalProbabilityCell(self.reverse_cell, reverse=True)
            self.total_prob_rnn = BaseRNN(self.total_prob_cell, batch_first=True, return_sequences=True, return_state=True)
            self.total_prob_rnn_rev = BaseRNN(
                self.total_prob_cell_rev,
                batch_first=True,
                return_sequences=True,
                return_state=True,
                reverse=True,
            )
        else:
            self.total_prob_rnn = None
            self.total_prob_rnn_rev = None

        self.built = True

    def create_cell(self):
        emitter_kwargs = dict(
            start_codons=self.start_codons,
            stop_codons=self.stop_codons,
            intron_begin_pattern=self.intron_begin_pattern,
            intron_end_pattern=self.intron_end_pattern,
            l2_lambda=self.variance_l2_lambda,
            num_models=self.num_models,
            num_copies=self.num_copies,
            init=self.emitter_init,
            trainable_emissions=self.trainable_emissions,
            emit_embeddings=self.emit_embeddings,
            embedding_dim=self.embedding_dim,
            full_covariance=self.full_covariance,
            embedding_kernel_init=self.embedding_kernel_init,
            initial_variance=self.initial_variance,
            temperature=self.temperature,
            share_intron_parameters=self.share_intron_parameters,
            trainable_nucleotides_at_exons=self.trainable_nucleotides_at_exons,
            device=self.device,
        )
        emitter = GenePredHMMEmitter(**emitter_kwargs)
        reverse_emitter = GenePredHMMEmitter(**emitter_kwargs)

        transitioner_kwargs = dict(
            k=self.num_copies,
            num_models=self.num_models,
            initial_exon_len=self.initial_exon_len,
            initial_intron_len=self.initial_intron_len,
            initial_ir_len=self.initial_ir_len,
            initial_utr_len=self.initial_utr_len,
            initial_utr_intron_len=self.initial_utr_intron_len,
            starting_distribution_init=self.starting_distribution_init,
            starting_distribution_trainable=self.trainable_starting_distribution,
            transitions_trainable=self.trainable_transitions,
            device=self.device,
        )
        transitioner = GenePredMultiHMMTransitioner(**transitioner_kwargs)
        reverse_transitioner = GenePredMultiHMMTransitioner(**transitioner_kwargs)
        reverse_transitioner.reverse = True

        cell = HmmCell(
            num_states=[emitter.num_states] * self.num_models,
            dim=self.dim,
            emitter=emitter,
            transitioner=transitioner,
            use_fake_step_counter=True,
            device=self.device,
        )
        reverse_cell = HmmCell(
            num_states=[reverse_emitter.num_states] * self.num_models,
            dim=self.dim,
            emitter=reverse_emitter,
            transitioner=reverse_transitioner,
            use_fake_step_counter=True,
            device=self.device,
        )
        reverse_cell.reverse = True
        return cell, reverse_cell

    @profile
    def forward(self, inputs, nucleotides=None, embeddings=None, end_hints=None, training=False, use_loglik=True):
        if end_hints is not None:
            end_hints = end_hints.unsqueeze(0)

        assert inputs.shape[-1] == self.dim, f"inputs should be of shape (batch, len, {self.dim})"
        assert nucleotides is not None and nucleotides.shape[-1] == 5, "nucleotides should be of shape (batch, len, 5)"

        stacked_inputs = self.concat_inputs(inputs, nucleotides, embeddings)

        if self.simple:
            log_post, prior, _ = self.state_posterior_log_probs(
                stacked_inputs,
                return_prior=True,
                end_hints=end_hints,
                training=training,
                no_loglik=not use_loglik,
            )
        else:
            log_post, prior, _ = _state_posterior_log_probs_impl(
                stacked_inputs,
                self.cell,
                self.reverse_cell,
                self.bidirectional_rnn,
                self.total_prob_rnn,
                self.total_prob_rnn_rev,
                end_hints=end_hints,
                return_prior=True,
                training=training,
                no_loglik=not use_loglik,
                parallel_factor=self.parallel_factor,
            )

        if training:
            prior = prior.mean()
            self.loss = prior
            if not hasattr(self, "metrics"):
                self.metrics = {}
            self.metrics["prior"] = prior.item()

        if self.num_models == 1:
            return log_post[0]
        return log_post.permute(1, 2, 0, 3)

    def concat_inputs(self, inputs, nucleotides, embeddings=None):
        inputs = inputs.unsqueeze(0)
        nucleotides = nucleotides.unsqueeze(0)
        chunks = [inputs]
        if self.emit_embeddings:
            assert embeddings is not None, "embeddings are required when emit_embeddings=True"
            chunks.append(embeddings.unsqueeze(0))
        chunks.append(nucleotides)
        return torch.cat(chunks, dim=-1)

    def viterbi(self, inputs, nucleotides, embeddings=None, end_hints=None):
        from .Viterbi import viterbi

        self.cell.recurrent_init()
        stacked_inputs = self.concat_inputs(inputs, nucleotides, embeddings)
        viterbi_seq = viterbi(stacked_inputs, self.cell, parallel_factor=self.parallel_factor)
        return viterbi_seq[0] if self.num_models == 1 else viterbi_seq.permute(1, 2, 0)
