"""Supported loss functions for OrionGeno training and fine-tuning."""

import torch
import torch.nn.functional as F


__all__ = [
    "compute_cross_entropy_loss",
    "compute_focal_cross_entropy_loss",
    "compute_weighted_token_cross_entropy_loss",
]


def compute_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Return mean token cross-entropy using PyTorch's default ignore index."""
    flat_logits = logits.contiguous().view(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    if not torch.any(flat_labels != -100):
        return logits.sum() * 0.0
    return F.cross_entropy(flat_logits, flat_labels)


def compute_weighted_token_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | list[float] | None = None,
    token_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Return token cross-entropy with optional class and per-token weights."""
    flat_logits = logits.contiguous().view(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    valid_mask = flat_labels != ignore_index
    flat_logits = flat_logits[valid_mask]
    flat_labels = flat_labels[valid_mask]

    if flat_labels.numel() == 0:
        return logits.sum() * 0.0

    if isinstance(class_weights, list):
        class_weights = torch.tensor(class_weights, device=logits.device)
    if class_weights is not None:
        class_weights = class_weights.to(device=logits.device, dtype=logits.dtype)

    per_token_loss = F.cross_entropy(
        flat_logits,
        flat_labels,
        weight=class_weights,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    if token_weights is None:
        return per_token_loss.mean()

    if token_weights.numel() != labels.numel():
        raise ValueError(
            "token_weights must contain one value per label: "
            f"got {token_weights.numel()} weights for {labels.numel()} labels"
        )

    flat_token_weights = token_weights.reshape(-1)[valid_mask].to(
        device=logits.device,
        dtype=per_token_loss.dtype,
    )
    weighted_loss = per_token_loss * flat_token_weights
    return weighted_loss.sum() / flat_token_weights.sum().clamp_min(1.0)


def compute_focal_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | list[float] | None = None,
    gamma: float = 2.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Return focal cross-entropy over all nonignored token labels."""
    if gamma < 0:
        raise ValueError(f"gamma must be nonnegative, got {gamma}")
    flat_logits = logits.contiguous().view(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    valid_mask = flat_labels != ignore_index
    flat_logits = flat_logits[valid_mask]
    flat_labels = flat_labels[valid_mask]

    if flat_labels.numel() == 0:
        return logits.sum() * 0.0

    if isinstance(class_weights, list):
        class_weights = torch.tensor(class_weights, device=logits.device)
    if class_weights is not None:
        class_weights = class_weights.to(device=logits.device, dtype=logits.dtype)

    per_token_cross_entropy = F.cross_entropy(
        flat_logits,
        flat_labels,
        weight=class_weights,
        reduction="none",
    )
    probabilities = F.softmax(flat_logits, dim=-1)
    target_probabilities = probabilities.gather(1, flat_labels.unsqueeze(1)).squeeze(1)
    focal_factor = (1.0 - target_probabilities).pow(gamma)
    return (focal_factor * per_token_cross_entropy).mean()
