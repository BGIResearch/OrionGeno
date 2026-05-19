import torch
import torch.nn as nn
from enum import Enum

from . import Initializers as initializers
from . import Priors as priors

class ProfileHMMEmitter(nn.Module):
    """Emission module for profile HMM states."""

    def __init__(self,
                 emission_init=initializers.make_default_emission_init(),
                 insertion_init=initializers.make_default_insertion_init(),
                 prior=None,
                 frozen_insertions=True,
                 device=None,
                 **kwargs):
        """Configure emission and insertion initializers for each profile model."""
        super(ProfileHMMEmitter, self).__init__(**kwargs)
        self.emission_init = [emission_init] if not hasattr(emission_init, '__iter__') else emission_init
        self.insertion_init = [insertion_init] if not hasattr(insertion_init, '__iter__') else insertion_init
        self.prior = priors.AminoAcidPrior() if prior is None else prior
        self.frozen_insertions = frozen_insertions

    def set_lengths(self, lengths):
        """Set per-model sequence lengths."""
        self.lengths = lengths
        self.num_models = len(lengths)

        assert len(self.lengths) == len(self.emission_init), \
            f"Expected one emission initializer per model, got {len(self.emission_init)} initializers for {len(self.lengths)} models."
        assert len(self.lengths) == len(self.insertion_init), \
            f"Expected one insertion initializer per model, got {len(self.insertion_init)} initializers for {len(self.lengths)} models."

    def build(self, input_shape):
        """Create module parameters lazily."""
        if hasattr(self, "built") and self.built:
            return
        s = input_shape[-1] - 1
        self.emission_kernel = nn.ParameterList([
            nn.Parameter(init(torch.Size([length, s])))
            for length, init in zip(self.lengths, self.emission_init)
        ])
        self.insertion_kernel = nn.ParameterList([
            nn.Parameter(init(torch.Size([s])))
            for init in self.insertion_init
        ])
        if self.frozen_insertions:
            for param in self.insertion_kernel:
                param.requires_grad = False
        self.prior.build()
        self.built = True

    def recurrent_init(self):
        """Prepare cached tensors used by recurrent evaluation."""
        self.B = self.make_B()
        self.B_transposed = torch.transpose(self.B, 1, 2)

    def make_emission_matrix(self, i):
        """Build the emission matrix for one model."""
        em, ins = self.emission_kernel[i], self.insertion_kernel[i]
        length = self.lengths[i]
        return self.make_emission_matrix_from_kernels(em, ins, length)

    def make_emission_matrix_from_kernels(self, em, ins, length):
        """Build an emission matrix from match and insertion kernels."""
        s = em.shape[-1]
        i1 = ins.unsqueeze(0)
        i2 = torch.stack([ins] * (length + 1))
        emissions = torch.cat([i1, em, i2], dim=0)
        emissions = torch.softmax(emissions, dim=-1)
        emissions = torch.cat([emissions, torch.zeros_like(emissions[:, :1])], dim=-1)
        end_state_emission = torch.nn.functional.one_hot(torch.tensor([s]), num_classes=s + 1, dtype=em.dtype)
        emissions = torch.cat([emissions, end_state_emission], dim=0)
        return emissions

    def make_B(self):
        """Build padded emission matrices for all models."""
        emission_matrices = []
        max_num_states = max([len(self.lengths) + 2] * self.num_models)
        for i in range(self.num_models):
            em_mat = self.make_emission_matrix(i)
            padding = max_num_states - em_mat.shape[0]
            em_mat_pad = torch.nn.functional.pad(em_mat, (0, 0, 0, padding))
            emission_matrices.append(em_mat_pad)
        B = torch.stack(emission_matrices, dim=0)
        return B

    def make_B_amino(self):
        """Return the emission matrix variant used by profile-HMM visualization."""
        return self.make_B()

    def forward(self, inputs, end_hints=None, training=False):
        """Run the module forward pass."""
        input_shape = inputs.shape
        inputs = inputs.view(inputs.shape[0], -1, input_shape[-1])
        B = self.B_transposed[..., :input_shape[-1], :]
        emit = torch.einsum("kbs,ksq->kbq", inputs, B)
        emit_shape = torch.Size([B.shape[0]] + list(input_shape[1:-1]) + [B.shape[-1]])
        emit = emit.view(emit_shape)
        return emit

    def get_aux_loss(self):
        """Return auxiliary regularization loss."""
        return torch.tensor(0., dtype=self.dtype)

    def get_prior_log_density(self):
        """Return the prior log density for current parameters."""
        return self.prior(self.B, lengths=self.lengths)

    def duplicate(self, model_indices=None, share_kernels=False):
        """Create a copy for a subset of models."""
        if model_indices is None:
            model_indices = range(len(self.emission_init))
        sub_emission_init = [initializers.ConstantInitializer(self.emission_kernel[i].detach().numpy()) for i in model_indices]
        sub_insertion_init = [initializers.ConstantInitializer(self.insertion_kernel[i].detach().numpy()) for i in model_indices]
        emitter_copy = ProfileHMMEmitter(
            emission_init=sub_emission_init,
            insertion_init=sub_insertion_init,
            prior=self.prior,
            frozen_insertions=self.frozen_insertions,
            dtype=self.dtype
        )
        if share_kernels:
            emitter_copy.emission_kernel = self.emission_kernel
            emitter_copy.insertion_kernel = self.insertion_kernel
            emitter_copy.built = True
        return emitter_copy

    def get_config(self):
        """Return serializable configuration values."""
        config = super(ProfileHMMEmitter, self).get_config()
        config.update({
            "lengths": self.lengths,
            "emission_init": self.emission_init,
            "insertion_init": self.insertion_init,
            "prior": self.prior,
            "frozen_insertions": self.frozen_insertions
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Create an instance from serialized configuration values."""
        config["emission_init"] = [initializers.deserialize(e) for e in config["emission_init"]]
        config["insertion_init"] = [initializers.deserialize(i) for i in config["insertion_init"]]
        config["prior"] = initializers.deserialize(config["prior"])
        lengths = config.pop("lengths")
        emitter = cls(**config)
        emitter.set_lengths(lengths)
        return emitter

    def __repr__(self):
        """Return a concise string representation."""
        return f"ProfileHMMEmitter(\n emission_init={self.emission_init[0]},\n insertion_init={self.insertion_init[0]},\n prior={self.prior},\n frozen_insertions={self.frozen_insertions}, )"

class TemperatureMode(Enum):
    TRAINABLE = 1
    LENGTH_NORM = 2
    COLD_TO_WARM = 3
    WARM_TO_COLD = 4
    CONSTANT = 5
    NONE = 6

    @staticmethod
    def from_string(name):
        return {"trainable": TemperatureMode.TRAINABLE,
                "length_norm": TemperatureMode.LENGTH_NORM,
                "cold_to_warm": TemperatureMode.COLD_TO_WARM,
                "warm_to_cold": TemperatureMode.WARM_TO_COLD,
                "constant": TemperatureMode.CONSTANT,
                "none": TemperatureMode.NONE}[name.lower()]
