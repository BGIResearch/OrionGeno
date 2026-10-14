from __future__ import annotations

from typing import Optional

import torch
from torch import nn


def prepare_frozen_species_embedding(
    species_embedding: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if species_embedding is None:
        return None
    if species_embedding.dim() == 2:
        species_embedding = species_embedding.unsqueeze(1)
    elif species_embedding.dim() == 3 and species_embedding.shape[1] == 1:
        pass
    elif species_embedding.dim() == 3 and species_embedding.shape[2] == 1:
        species_embedding = species_embedding.transpose(1, 2)
    else:
        raise ValueError(
            "species_embedding must have shape [batch, channels] or [batch, 1, channels]"
        )
    return species_embedding.detach()


class SpeciesConditionProjector(nn.Module):
    """Project a frozen species embedding into one injection point's channels.

    The species embedding stays frozen (detached upstream). The projected
    condition is LayerNorm-stabilized and scaled by a learnable gain.

    Output is channel-first ``[B, C, 1]`` to broadcast over the sequence length,
    matching the legacy additive injection.
    """

    def __init__(self, embedding_dim: int, channels: int, gain_init: float = 0.3):
        super().__init__()
        self.proj = nn.Linear(embedding_dim, channels)
        self.norm = nn.LayerNorm(channels)
        self.gain = nn.Parameter(torch.tensor(float(gain_init)))

    def forward(self, species_embedding: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        frozen = prepare_frozen_species_embedding(species_embedding)
        if frozen is None:
            return None
        # frozen: [B, 1, emb_dim] -> projected [B, 1, C] -> channel-first [B, C, 1]
        projected = self.norm(self.proj(frozen))
        condition = (self.gain * projected).permute(0, 2, 1)
        return condition


def build_species_condition(
    species_embedding: Optional[torch.Tensor],
    *,
    expected_channels: int,
    context: str,
) -> Optional[torch.Tensor]:
    """Legacy raw-additive condition: requires the embedding to already be the
    target width. Kept for backward compatibility with old per-layer-sized
    embeddings; the projected-gain path (SpeciesConditionProjector) is preferred.
    """
    frozen_embedding = prepare_frozen_species_embedding(species_embedding)
    if frozen_embedding is None:
        return None

    channel_first_condition = frozen_embedding.permute(0, 2, 1)
    if channel_first_condition.shape[1] != expected_channels:
        raise ValueError(
            f"{context} expects species_embedding width {expected_channels}, "
            f"got {channel_first_condition.shape[1]}"
        )
    return channel_first_condition
