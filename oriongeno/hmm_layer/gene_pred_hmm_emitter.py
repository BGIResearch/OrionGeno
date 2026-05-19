import torch
import torch.nn as nn
import torch.nn.functional as F

from . import kmer
from .Initializers import ConstantInitializer, make_20_class_emission_kernel
from .MvnMixture import DefaultDiagBijector, MvnMixture
from .gene_pred_hmm_topology import (
    BASE_GENE_PRED_STATE_TO_INDEX,
    expanded_num_states,
    expanded_state_index,
)


def assert_codons(codons):
    total = sum(prob for _, prob in codons)
    assert abs(total - 1.0) < 1e-6, f"Codon probabilities must sum to 1, got {codons}"
    for pattern, prob in codons:
        assert len(pattern) == 3, f"Expected 3-mer pattern, got {pattern}"
        assert 0.0 <= prob <= 1.0, f"Invalid probability {prob} in {codons}"


def make_codon_probs(codons, pivot_left):
    assert_codons(codons)
    codon_probs = sum(
        prob * kmer.encode_kmer_string(pattern, pivot_left)
        for pattern, prob in codons
    )
    return codon_probs.reshape(1, 1, 64)


class SimpleGenePredHMMEmitter(nn.Module):
    def __init__(
        self,
        num_models=1,
        num_copies=1,
        init=None,
        trainable_emissions=False,
        emit_embeddings=False,
        embedding_dim=None,
        full_covariance=False,
        embedding_kernel_init="random_normal",
        initial_variance=0.05,
        temperature=100.0,
        share_intron_parameters=False,
        device=None,
        **kwargs,
    ):
        super(SimpleGenePredHMMEmitter, self).__init__(**kwargs)
        if init is None:
            init = make_20_class_emission_kernel(
                smoothing=1e-2,
                num_copies=num_copies,
                num_models=num_models,
            )

        self.num_models = num_models
        self.num_copies = num_copies
        self.init = init
        self.num_states = int(init.shape[1])
        self.label_dim = int(init.shape[-1])
        self.trainable_emissions = trainable_emissions
        self.emit_embeddings = emit_embeddings
        self.embedding_dim = embedding_dim
        self.full_covariance = full_covariance
        self.embedding_kernel_init = embedding_kernel_init
        self.initial_variance = initial_variance
        self.temperature = temperature
        self.share_intron_parameters = share_intron_parameters
        self.device = device

        if self.emit_embeddings:
            assert embedding_dim is not None, "embedding_dim is required when emit_embeddings=True"
        else:
            assert embedding_dim is None, "embedding_dim must be None when emit_embeddings=False"

        self.emission_kernel = None
        self.embedding_emission_kernel = None
        self.mvn_mixture = None
        self.B = None
        self.embedding_emit = None
        self.built = False

    def build(self):
        if self.built:
            return
        self.emission_kernel = nn.Parameter(
            torch.as_tensor(self.init, dtype=torch.float32),
            requires_grad=self.trainable_emissions,
        )
        if self.emit_embeddings:
            assert self.num_models == 1, "Embedding emissions currently support num_models == 1 only."
            d = self.embedding_dim
            num_mvn_param = d + d * (d + 1) // 2 if self.full_covariance else 2 * d
            if self.embedding_kernel_init != "random_normal":
                raise ValueError(f"embedding_kernel_init '{self.embedding_kernel_init}' not supported")
            self.embedding_emission_kernel = nn.Parameter(
                torch.randn(1, self.num_states, 1, num_mvn_param),
                requires_grad=True,
            )
        self.built = True

    def recurrent_init(self):
        self.B = self.make_B()
        if self.emit_embeddings:
            self.mvn_mixture = MvnMixture(
                self.embedding_dim,
                self.embedding_emission_kernel,
                diag_only=not self.full_covariance,
                diag_bijector=DefaultDiagBijector(self.initial_variance),
            )

    def make_B(self):
        return F.softmax(self.emission_kernel, dim=-1)

    def apply_state_parameter_sharing(self, emit):
        return emit

    def forward(self, inputs, end_hints=None, training=False):
        if self.emit_embeddings:
            class_inputs = inputs[..., :-self.embedding_dim]
            embedding_inputs = inputs[..., -self.embedding_dim:]
            class_emit = torch.einsum(
                "kbls,kqs->kblq",
                class_inputs,
                self.B.to(device=inputs.device, dtype=inputs.dtype),
            )
            embedding_inputs = embedding_inputs.view(1, -1, self.embedding_dim)
            log_pdf = self.mvn_mixture.log_pdf(embedding_inputs)
            log_pdf = log_pdf.view(class_emit.shape)
            self.embedding_emit = torch.exp(log_pdf / self.temperature)
            if training:
                class_emit = class_emit + 1e-10
                self.embedding_emit = self.embedding_emit + 1e-10
            emit = class_emit * self.embedding_emit
        else:
            emit = torch.einsum(
                "kbls,kqs->kblq",
                inputs,
                self.B.to(device=inputs.device, dtype=inputs.dtype),
            )

        if self.share_intron_parameters:
            emit = self.apply_state_parameter_sharing(emit)

        if end_hints is not None:
            left_end = end_hints[..., :1, :] * emit[..., :1, :]
            right_end = end_hints[..., 1:, :] * emit[..., -1:, :]
            emit = torch.cat([left_end, emit[..., 1:-1, :], right_end], dim=-2)
        return emit

    def get_prior_log_density(self):
        return torch.tensor([[0.0]])

    def get_aux_loss(self):
        return torch.tensor(0.0)

    def get_config(self):
        return {
            "num_models": self.num_models,
            "num_copies": self.num_copies,
            "init": self.init,
            "trainable_emissions": self.trainable_emissions,
            "emit_embeddings": self.emit_embeddings,
            "embedding_dim": self.embedding_dim,
            "full_covariance": self.full_covariance,
            "embedding_kernel_init": self.embedding_kernel_init,
            "initial_variance": self.initial_variance,
            "temperature": self.temperature,
            "share_intron_parameters": self.share_intron_parameters,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class GenePredHMMEmitter(SimpleGenePredHMMEmitter):
    def __init__(
        self,
        start_codons,
        stop_codons,
        intron_begin_pattern,
        intron_end_pattern,
        l2_lambda=0.01,
        nucleotide_kernel_init=ConstantInitializer(0.0),
        trainable_nucleotides_at_exons=False,
        **kwargs,
    ):
        super(GenePredHMMEmitter, self).__init__(**kwargs)
        self.num_states = expanded_num_states(self.num_copies)
        self.start_codons = start_codons
        self.stop_codons = stop_codons
        self.intron_begin_pattern = intron_begin_pattern
        self.intron_end_pattern = intron_end_pattern
        self.l2_lambda = l2_lambda
        self.nucleotide_kernel_init = nucleotide_kernel_init
        self.trainable_nucleotides_at_exons = trainable_nucleotides_at_exons
        self.exon_state_indices = [
            expanded_state_index(base_state, copy_index)
            for copy_index in range(self.num_copies)
            for base_state in (4, 5, 6)
        ]
        self.intron_sharing_groups = [
            (
                expanded_state_index(1, copy_index),
                [
                    expanded_state_index(2, copy_index),
                    expanded_state_index(3, copy_index),
                ],
            )
            for copy_index in range(self.num_copies)
        ]
        self.codon_probs = nn.Parameter(
            self._build_codon_probs(),
            requires_grad=False,
        )
        self.build()

    def _build_codon_probs(self):
        state = BASE_GENE_PRED_STATE_TO_INDEX
        any_codon = make_codon_probs([("NNN", 1.0)], pivot_left=False)
        not_stop = any_codon * (make_codon_probs(self.stop_codons, pivot_left=False) == 0).float()
        not_stop = not_stop / not_stop.sum().clamp_min(1e-12)
        start = make_codon_probs(self.start_codons, pivot_left=True)
        donor = make_codon_probs(self.intron_begin_pattern, pivot_left=True)
        acceptor = make_codon_probs(self.intron_end_pattern, pivot_left=False)

        num_states = expanded_num_states(self.num_copies)
        left = [any_codon for _ in range(num_states)]
        right = [any_codon for _ in range(num_states)]

        for copy_index in range(self.num_copies):
            left[expanded_state_index(state["START"], copy_index)] = start
            right[expanded_state_index(state["Exon2"], copy_index)] = not_stop
            right[expanded_state_index(state["STOP"], copy_index)] = make_codon_probs(
                self.stop_codons,
                pivot_left=False,
            )
            for base_state in ("EI0", "EI1", "EI2"):
                left[expanded_state_index(state[base_state], copy_index)] = donor
            for base_state in ("IE0", "IE1", "IE2"):
                right[expanded_state_index(state[base_state], copy_index)] = acceptor

        left_probs = torch.cat(left, dim=1)
        right_probs = torch.cat(right, dim=1)
        return torch.cat([left_probs, right_probs], dim=0)

    def build(self):
        if self.built:
            return
        super(GenePredHMMEmitter, self).build()
        if self.trainable_nucleotides_at_exons:
            assert self.num_models == 1, "Trainable nucleotide emissions currently support num_models == 1 only."
            self.nuc_emission_kernel = nn.Parameter(
                torch.zeros(self.num_models, 3 * self.num_copies, 4),
                requires_grad=True,
            )

    def apply_state_parameter_sharing(self, emit):
        if not self.share_intron_parameters:
            return emit
        emit = emit.clone()
        for canonical_state, shared_states in self.intron_sharing_groups:
            emit[..., shared_states] = emit[..., canonical_state].unsqueeze(-1)
        return emit

    def get_nucleotide_probs(self):
        return torch.softmax(self.nuc_emission_kernel, dim=-1)

    def forward(self, inputs, end_hints=None, training=False):
        nucleotides = inputs[..., -5:]
        class_inputs = inputs[..., :-5]
        emit = super(GenePredHMMEmitter, self).forward(class_inputs, end_hints=None, training=training)

        num_models, batch, length = nucleotides.shape[:3]
        flat_nucleotides = nucleotides.reshape(-1, length, 5)
        left_3mers = kmer.make_k_mers(flat_nucleotides, k=3, pivot_left=True).reshape(
            num_models,
            batch,
            length,
            64,
        )
        right_3mers = kmer.make_k_mers(flat_nucleotides, k=3, pivot_left=False).reshape(
            num_models,
            batch,
            length,
            64,
        )
        input_3mers = torch.stack([left_3mers, right_3mers], dim=-2)
        codon_emission = torch.einsum(
            "kblrs,rqs->kblrq",
            input_3mers,
            self.codon_probs.to(device=inputs.device, dtype=inputs.dtype),
        ).prod(dim=-2)
        full_emission = emit * codon_emission

        if self.trainable_nucleotides_at_exons:
            nucleotides_no_n = nucleotides[..., :4] + nucleotides[..., 4:] / 4.0
            exon_emission = torch.einsum(
                "kbls,kqs->kblq",
                nucleotides_no_n,
                self.get_nucleotide_probs().to(device=inputs.device, dtype=inputs.dtype),
            )
            exon_mask = torch.ones_like(full_emission)
            exon_mask[..., self.exon_state_indices] = exon_emission
            full_emission = full_emission * exon_mask

        if training:
            full_emission = full_emission + 1e-7

        if end_hints is not None:
            left_end = end_hints[..., :1, :] * full_emission[..., :1, :]
            right_end = end_hints[..., 1:, :] * full_emission[..., -1:, :]
            full_emission = torch.cat([left_end, full_emission[..., 1:-1, :], right_end], dim=-2)

        if self.emit_embeddings:
            self.loss = self.l2_lambda * self.mvn_mixture.get_regularization_L2_loss()

        return full_emission

    def duplicate(self, model_indices=None, share_kernels=False):
        init = self.emission_kernel.detach().cpu().numpy()
        emitter_copy = GenePredHMMEmitter(
            start_codons=self.start_codons,
            stop_codons=self.stop_codons,
            intron_begin_pattern=self.intron_begin_pattern,
            intron_end_pattern=self.intron_end_pattern,
            l2_lambda=self.l2_lambda,
            nucleotide_kernel_init=self.nucleotide_kernel_init,
            num_models=self.num_models,
            num_copies=self.num_copies,
            init=init,
            trainable_emissions=self.trainable_emissions,
            emit_embeddings=self.emit_embeddings,
            embedding_dim=self.embedding_dim,
            full_covariance=self.full_covariance,
            embedding_kernel_init=self.embedding_kernel_init,
            initial_variance=self.initial_variance,
            temperature=self.temperature,
            share_intron_parameters=self.share_intron_parameters,
            trainable_nucleotides_at_exons=self.trainable_nucleotides_at_exons,
        )
        if share_kernels:
            emitter_copy.emission_kernel = self.emission_kernel
            if self.trainable_nucleotides_at_exons:
                emitter_copy.nuc_emission_kernel = self.nuc_emission_kernel
            if self.emit_embeddings:
                emitter_copy.embedding_emission_kernel = self.embedding_emission_kernel
            emitter_copy.built = True
        return emitter_copy

    def get_config(self):
        config = super(GenePredHMMEmitter, self).get_config()
        config.update(
            {
                "start_codons": self.start_codons,
                "stop_codons": self.stop_codons,
                "intron_begin_pattern": self.intron_begin_pattern,
                "intron_end_pattern": self.intron_end_pattern,
                "l2_lambda": self.l2_lambda,
                "nucleotide_kernel_init": self.nucleotide_kernel_init,
                "trainable_nucleotides_at_exons": self.trainable_nucleotides_at_exons,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
