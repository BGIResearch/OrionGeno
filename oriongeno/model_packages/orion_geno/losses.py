import torch
import torch.nn.functional as F


def compute_cross_entropy_loss(logits, labels):
    logits = logits.contiguous().view(-1, logits.shape[-1])
    labels = labels.view(-1)
    return F.cross_entropy(logits, labels)


def compute_weighted_token_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | list[float] | None = None,
    token_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    flat_logits = logits.contiguous().view(-1, logits.shape[-1])
    flat_labels = labels.view(-1)
    valid_mask = flat_labels != ignore_index
    flat_logits = flat_logits[valid_mask]
    flat_labels = flat_labels[valid_mask]

    if flat_labels.numel() == 0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)

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
    flat_logits = logits.contiguous().view(-1, logits.shape[-1])
    flat_labels = labels.view(-1)
    valid_mask = flat_labels != ignore_index
    flat_logits = flat_logits[valid_mask]
    flat_labels = flat_labels[valid_mask]

    if flat_labels.numel() == 0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)

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
