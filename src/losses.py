"""Partial (masked) cross-entropy and partial focal loss for point supervision."""

from __future__ import annotations

import torch
import torch.nn.functional as F


IGNORE = 255


def partial_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Mean CE only where mask is True.

    logits: [B, C, H, W]
    target: [B, H, W] int64 class ids (any value at ignored positions; masked out)
    mask: [B, H, W] bool — True at labeled point pixels
    """
    t = target.clone()
    t[~mask] = IGNORE
    ce = F.cross_entropy(logits, t, reduction="none", ignore_index=IGNORE)
    m = mask.float()
    denom = m.sum().clamp(min=eps)
    return (ce * m).sum() / denom


def partial_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    pfCE = sum(Focal(pred, GT) * MASK) / sum(MASK)  (per batch scalar; masked positions excluded).

    Multiclass focal: (1 - p_t)^gamma * CE, averaged over labeled pixels only.
    """
    t = target.clone()
    t[~mask] = IGNORE
    ce = F.cross_entropy(logits, t, reduction="none", ignore_index=IGNORE)

    p = F.softmax(logits, dim=1)
    # True-class probability at each pixel (invalid positions multiplied out later)
    lg = target.clone()
    lg[~mask] = 0
    p_t = p.gather(1, lg.unsqueeze(1)).squeeze(1).clamp(min=eps, max=1.0 - eps)
    focal = (1.0 - p_t).pow(gamma) * ce
    m = mask.float()
    denom = m.sum().clamp(min=eps)
    return (focal * m).sum() / denom
