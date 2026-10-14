import math
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_orion_geno import OrionGenoConfig
from .sequence_pyramid import (
    SequencePyramidDecoder,
    SequencePyramidEncoder,
    encode_tokens_as_one_hot,
)
from .species_conditioning import prepare_frozen_species_embedding

try:
    from mamba_ssm.modules.block import Block
    from mamba_ssm.modules.mamba2 import Mamba2
    from mamba_ssm.modules.mamba2_simple import Mamba2Simple
    from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.modules.mha import MHA
    from mamba_ssm.modules.mlp import GatedMLP
except ImportError as import_error:
    Block = None
    Mamba2 = None
    Mamba2Simple = None
    Mamba = None
    MHA = None
    GatedMLP = None
    _MAMBA_SSM_IMPORT_ERROR = import_error
else:
    _MAMBA_SSM_IMPORT_ERROR = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm = None
    layer_norm_fn = None
    rms_norm_fn = None


@dataclass
class OrionGenoOutput(ModelOutput):
    logits: Optional[Tuple[torch.Tensor, ...]] = None
    gene_logits: Optional[torch.Tensor] = None
    repeat_logits: Optional[torch.Tensor] = None
    boundary_logits: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None


def _require_mamba_ssm_dependencies() -> None:
    if all(
        dependency is not None
        for dependency in (Block, Mamba2, Mamba2Simple, Mamba, MHA, GatedMLP)
    ):
        return
    raise ImportError(
        "OrionGeno's sequence backbone requires the `mamba_ssm` "
        "package. Install a compatible `mamba_ssm` build before training or "
        "loading OrionGeno checkpoints."
    ) from _MAMBA_SSM_IMPORT_ERROR


def _resolve_state_space_layer(ssm_layer_name: str):
    _require_mamba_ssm_dependencies()
    if ssm_layer_name == "Mamba2":
        return Mamba2
    if ssm_layer_name == "Mamba2Simple":
        return Mamba2Simple
    if ssm_layer_name == "Mamba1":
        return Mamba
    raise ValueError(
        f"Unsupported sequence state-space layer: {ssm_layer_name}. "
        "Expected one of Mamba1, Mamba2, or Mamba2Simple."
    )


class DirectionalSequenceMixer(nn.Module):
    def __init__(
        self,
        d_model: int,
        ssm_layer_type: str = "Mamba2",
        bidirectional: bool = True,
        bidirectional_strategy: Optional[str] = "add",
        bidirectional_weight_tie: bool = True,
        **mamba_kwargs,
    ):
        super().__init__()
        _require_mamba_ssm_dependencies()

        if bidirectional and bidirectional_strategy is None:
            bidirectional_strategy = "add"
        if bidirectional and bidirectional_strategy not in ("add", "ew_multiply"):
            raise NotImplementedError(
                f"Unsupported bidirectional strategy: {bidirectional_strategy}"
            )

        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        self.ssm_layer_type = _resolve_state_space_layer(ssm_layer_type)
        self.mamba_fwd = self.ssm_layer_type(d_model=d_model, **mamba_kwargs)
        if bidirectional:
            self.mamba_rev = self.ssm_layer_type(d_model=d_model, **mamba_kwargs)
            if bidirectional_weight_tie:
                self.mamba_rev.in_proj.weight = self.mamba_fwd.in_proj.weight
                self.mamba_rev.in_proj.bias = self.mamba_fwd.in_proj.bias
                self.mamba_rev.out_proj.weight = self.mamba_fwd.out_proj.weight
                self.mamba_rev.out_proj.bias = self.mamba_fwd.out_proj.bias
        else:
            self.mamba_rev = None

    def forward(self, hidden_states, inference_params=None):
        output = self.mamba_fwd(hidden_states, inference_params=inference_params)
        if self.bidirectional:
            reverse_output = self.mamba_rev(
                hidden_states.flip(dims=(1,)),
                inference_params=inference_params,
            ).flip(dims=(1,))
            if self.bidirectional_strategy == "add":
                output = output + reverse_output
            elif self.bidirectional_strategy == "ew_multiply":
                output = output * reverse_output
            else:
                raise NotImplementedError(
                    f"Unsupported bidirectional strategy: {self.bidirectional_strategy}"
                )
        return output


def build_sequence_mixer_block(
    d_model: int,
    d_intermediate: int,
    ssm_cfg=None,
    attn_layer_idx=None,
    attn_cfg=None,
    attention_type: str = "flash_attention_2",
    norm_epsilon: float = 1e-5,
    rms_norm: bool = False,
    residual_in_fp32: bool = False,
    fused_add_norm: bool = False,
    layer_idx: Optional[int] = None,
    bidirectional: bool = False,
    bidirectional_strategy: str = "add",
    bidirectional_weight_tie: bool = True,
    device=None,
    dtype=None,
):
    _require_mamba_ssm_dependencies()
    if rms_norm and RMSNorm is None:
        raise ImportError(
            "OrionGeno is configured with rms_norm=True, but mamba_ssm's RMSNorm "
            "implementation is unavailable in the current environment."
        )
    ssm_cfg = dict(ssm_cfg or {})
    attn_layer_idx = list(attn_layer_idx or [])
    attn_cfg = dict(attn_cfg or {})
    factory_kwargs = {"device": device, "dtype": dtype}

    if layer_idx not in attn_layer_idx:
        ssm_layer_name = ssm_cfg.pop("layer", "Mamba1")
        if bidirectional:
            mixer_cls = partial(
                DirectionalSequenceMixer,
                ssm_layer_type=ssm_layer_name,
                layer_idx=layer_idx,
                bidirectional=bidirectional,
                bidirectional_strategy=bidirectional_strategy,
                bidirectional_weight_tie=bidirectional_weight_tie,
                **ssm_cfg,
                **factory_kwargs,
            )
        else:
            mixer_cls = partial(
                _resolve_state_space_layer(ssm_layer_name),
                layer_idx=layer_idx,
                **ssm_cfg,
                **factory_kwargs,
            )
    else:
        if attention_type != "flash_attention_2":
            raise NotImplementedError(
                "Only flash_attention_2 attention blocks are supported."
            )
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)

    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm,
        eps=norm_epsilon,
        **factory_kwargs,
    )
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP,
            hidden_features=d_intermediate,
            out_features=d_model,
            **factory_kwargs,
        )

    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


class OrionSequenceModel(nn.Module):
    def __init__(self, config: OrionGenoConfig, device=None, dtype=None):
        super().__init__()
        _require_mamba_ssm_dependencies()
        if config.rms_norm and RMSNorm is None:
            raise ImportError(
                "OrionGeno is configured with rms_norm=True, but mamba_ssm's RMSNorm "
                "implementation is unavailable in the current environment."
            )

        factory_kwargs = {"device": device, "dtype": dtype}
        self.fused_add_norm = config.fused_add_norm
        self.residual_in_fp32 = config.residual_in_fp32
        if self.fused_add_norm and (layer_norm_fn is None or rms_norm_fn is None):
            raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                build_sequence_mixer_block(
                    config.d_model,
                    d_intermediate=config.d_intermediate,
                    ssm_cfg=config.ssm_cfg,
                    attn_layer_idx=config.attn_layer_idx,
                    attn_cfg=config.attn_cfg,
                    attention_type=getattr(
                        config,
                        "attention_type",
                        "flash_attention_2",
                    ),
                    norm_epsilon=config.norm_epsilon,
                    rms_norm=config.rms_norm,
                    residual_in_fp32=config.residual_in_fp32,
                    fused_add_norm=config.fused_add_norm,
                    layer_idx=layer_index,
                    bidirectional=config.bidirectional,
                    bidirectional_strategy=config.bidirectional_strategy,
                    bidirectional_weight_tie=config.bidirectional_weight_tie,
                    **factory_kwargs,
                )
                for layer_index in range(config.n_layer)
            ]
        )
        self.output_norm = (nn.LayerNorm if not config.rms_norm else RMSNorm)(
            config.d_model,
            eps=config.norm_epsilon,
            **factory_kwargs,
        )

    def forward(
        self,
        input_embeddings: torch.Tensor,
        output_hidden_states: bool = False,
        inference_params=None,
        **mixer_kwargs,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        all_hidden_states = []
        residual, hidden_states = None, input_embeddings
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            hidden_states, residual = layer(
                hidden_states,
                residual,
                inference_params=inference_params,
                **mixer_kwargs,
            )

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.output_norm(
                residual.to(dtype=self.output_norm.weight.dtype)
            )
        else:
            uses_rms_norm = RMSNorm is not None and isinstance(self.output_norm, RMSNorm)
            fused_add_norm_function = (
                rms_norm_fn if uses_rms_norm else layer_norm_fn
            )
            hidden_states = fused_add_norm_function(
                hidden_states,
                self.output_norm.weight,
                getattr(self.output_norm, "bias", None),
                eps=self.output_norm.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        return hidden_states, all_hidden_states


class OrionGenoBackbone(nn.Module):
    def __init__(self, config: OrionGenoConfig):
        super().__init__()
        settings = getattr(config, "settings", {})
        self.use_decoder_skip_connection = bool(
            settings.get("use_decoder_skip_connection", False)
        )
        self.encoder = SequencePyramidEncoder(config)
        self.sequence_model = OrionSequenceModel(config)
        self.decoder = SequencePyramidDecoder(config)

    def forward(
        self,
        sequence_features: torch.Tensor,
        species_embedding: Optional[torch.Tensor] = None,
        species_group_ids: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        del species_group_ids
        encoded_features, skip_connections = self.encoder(
            sequence_features,
            species_embedding=species_embedding,
        )
        sequence_features, all_hidden_states = self.sequence_model(
            encoded_features,
            output_hidden_states=output_hidden_states,
        )
        decoded_features = self.decoder(
            sequence_features,
            skip_connections,
            species_embedding=species_embedding,
        )

        if self.use_decoder_skip_connection:
            if decoded_features.shape != encoded_features.shape:
                raise ValueError(
                    "use_decoder_skip_connection requires decoder and encoder features "
                    "to share the same shape."
                )
            decoded_features = decoded_features + encoded_features

        return decoded_features, all_hidden_states


class OrionGenoPreTrainedModel(PreTrainedModel):
    config_class = OrionGenoConfig
    base_model_prefix = "backbone"
    supports_gradient_checkpointing = False

    def __init__(self, config: OrionGenoConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (
                config.vocab_size % config.pad_vocab_size_multiple
            )
        config_dtype = getattr(config, "dtype", None)
        self.model_dtype = (
            getattr(torch, config_dtype)
            if isinstance(config_dtype, str)
            else config_dtype
        )
        if self.model_dtype is None:
            self.model_dtype = torch.float32

    def _init_weights(self, module, initializer_range: float = 0.02, **kwargs):
        del kwargs
        initializer_config = (
            self.config.initializer_cfg if self.config.initializer_cfg is not None else {}
        )
        rescale_prenorm_residual = initializer_config.get(
            "rescale_prenorm_residual",
            True,
        )
        initializer_range = initializer_config.get(
            "initializer_range",
            initializer_range,
        )
        n_residuals_per_layer = initializer_config.get("n_residuals_per_layer", 1)

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None and not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=initializer_range)
        elif isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            module.weight.data.normal_(mean=0.0, std=initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

        if rescale_prenorm_residual:
            for name, parameter in module.named_parameters():
                if name in ("out_proj.weight", "fc2.weight"):
                    nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))
                    with torch.no_grad():
                        parameter /= math.sqrt(
                            n_residuals_per_layer * self.config.n_layer
                        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        return_tuple = kwargs.pop("output_loading_info", False)
        loaded = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            output_loading_info=return_tuple,
            **kwargs,
        )
        if return_tuple:
            model, loading_info = loaded
        else:
            model, loading_info = loaded, None
        model._restore_weight_tying()
        if return_tuple:
            return model, loading_info
        return model

    def _restore_weight_tying(self) -> None:
        backbone = getattr(self, "backbone", None)
        sequence_model = getattr(backbone, "sequence_model", None)
        if sequence_model is None:
            return
        for layer in sequence_model.layers:
            mixer = getattr(layer, "mixer", None)
            if hasattr(mixer, "mamba_rev") and mixer.mamba_rev is not None:
                mixer.mamba_rev.in_proj.weight = mixer.mamba_fwd.in_proj.weight
                mixer.mamba_rev.in_proj.bias = mixer.mamba_fwd.in_proj.bias
                mixer.mamba_rev.out_proj.weight = mixer.mamba_fwd.out_proj.weight
                mixer.mamba_rev.out_proj.bias = mixer.mamba_fwd.out_proj.bias


class OrionGenoForJointLabeling(OrionGenoPreTrainedModel):
    def __init__(self, config: OrionGenoConfig):
        super().__init__(config)
        settings = getattr(config, "settings", {})
        input_channels = settings.get("input_channels", 5)
        decoder_channel_sizes = settings.get(
            "decoder_channel_sizes",
            [768, 768, 640, 512, 384],
        )
        decoder_output_channels = decoder_channel_sizes[-1]

        self.input_channels = input_channels
        self.gene_label_count = int(settings.get("gene_label_count", 20))
        self.repeat_label_count = int(settings.get("repeat_label_count", 2))
        self.backbone = OrionGenoBackbone(config)
        self.gene_classifier = nn.Linear(
            decoder_output_channels,
            self.gene_label_count,
        )
        self.repeat_classifier = nn.Linear(
            decoder_output_channels,
            self.repeat_label_count,
        )
        self.boundary_classifier = nn.Linear(decoder_output_channels, 2)
        self.post_init()
        self._restore_weight_tying()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        species_embedding: Optional[torch.Tensor] = None,
        species_group_ids: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, ...], OrionGenoOutput]:
        del attention_mask

        return_dict = (
            self.config.use_return_dict if return_dict is None else return_dict
        )
        output_hidden_states = (
            self.config.output_hidden_states
            if output_hidden_states is None
            else output_hidden_states
        )
        species_embedding = prepare_frozen_species_embedding(species_embedding)
        del species_group_ids

        sequence_features = encode_tokens_as_one_hot(
            input_ids,
            vocab_size=self.config.vocab_size,
            output_size=self.input_channels,
            pad_id=self.config.pad_token_id,
            mask_id=getattr(self.config, "mask_token_id", 10),
            dtype=self.gene_classifier.weight.dtype,
        )

        decoder_features, all_hidden_states = self.backbone(
            sequence_features,
            species_embedding=species_embedding,
            output_hidden_states=output_hidden_states,
        )
        gene_logits = self.gene_classifier(decoder_features)
        repeat_logits = self.repeat_classifier(decoder_features)
        boundary_logits = self.boundary_classifier(decoder_features)

        if not return_dict:
            return (
                gene_logits,
                repeat_logits,
                boundary_logits,
                decoder_features,
                all_hidden_states,
            )

        return OrionGenoOutput(
            logits=(gene_logits, repeat_logits, boundary_logits),
            gene_logits=gene_logits,
            repeat_logits=repeat_logits,
            boundary_logits=boundary_logits,
            hidden_states=decoder_features,
        )
