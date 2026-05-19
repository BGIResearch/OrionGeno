from typing import Any, Dict, List, Optional

from transformers import PretrainedConfig


class OrionGenoConfig(PretrainedConfig):
    model_type = "orion_geno"

    def __init__(
        self,
        d_model: int = 4,
        n_layer: int = 2,
        d_intermediate: int = 0,
        vocab_size: int = 50277,
        ssm_cfg: Optional[dict] = None,
        attn_cfg: Optional[dict] = None,
        attn_layer_idx: Optional[List[int]] = None,
        attention_type: Optional[str] = "flash_attention_2",
        rms_norm: bool = True,
        residual_in_fp32: bool = True,
        fused_add_norm: bool = True,
        pad_vocab_size_multiple: int = 1,
        norm_epsilon: float = 1e-5,
        initializer_cfg: Optional[dict] = None,
        loss_weights: Optional[List] = None,
        pad_token_id: int = 9,
        bos_token_id: int = 11,
        eos_token_id: int = 12,
        mask_token_id: int = 10,
        bidirectional: bool = False,
        bidirectional_strategy: str = "add",
        bidirectional_weight_tie: bool = True,
        use_second_lm_head: bool = False,
        dtype: Optional[str] = None,
        settings: Optional[Dict] = None,
        *args,
        **kwargs,
    ):
        retired_settings = kwargs.pop("configs", None)
        if settings is None and retired_settings is not None:
            raise ValueError(
                "OrionGenoConfig no longer supports the retired 'configs' field. "
                "Provide model settings under 'settings' instead."
            )
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.d_model = d_model
        self.n_layer = n_layer
        self.d_intermediate = d_intermediate
        self.vocab_size = vocab_size
        self.ssm_cfg = ssm_cfg or {}
        self.attn_cfg = attn_cfg or {}
        self.attn_layer_idx = list(attn_layer_idx or [])
        self.attention_type = attention_type
        self.rms_norm = rms_norm
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.norm_epsilon = norm_epsilon
        self.initializer_cfg = initializer_cfg
        self.loss_weights = loss_weights
        self.mask_token_id = mask_token_id
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        self.bidirectional_weight_tie = bidirectional_weight_tie
        self.use_second_lm_head = use_second_lm_head
        self.dtype = dtype

        self.settings = dict(settings or {})

    def to_dict(self) -> Dict[str, Any]:
        output = super().to_dict()
        output["settings"] = dict(self.settings)
        return output
