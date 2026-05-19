from typing import Optional

import torch


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


def build_species_condition(
    species_embedding: Optional[torch.Tensor],
    *,
    expected_channels: int,
    context: str,
) -> Optional[torch.Tensor]:
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
