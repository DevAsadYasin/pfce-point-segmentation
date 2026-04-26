"""Full-mask segmentation metrics (mIoU, per-class IoU)."""

from __future__ import annotations

import torch


def confusion_matrix(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = 0,
) -> torch.Tensor:
    """Accumulate [C, C] confusion for argmax predictions vs target."""
    pred = logits.argmax(dim=1)
    mask = torch.ones_like(target, dtype=torch.bool)
    if ignore_index is not None:
        mask &= target != ignore_index
    pred = pred[mask]
    tgt = target[mask]
    idx = tgt * num_classes + pred
    cm = torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm


def miou_from_confusion(cm: torch.Tensor, ignore_index: int | None = 0) -> tuple[float, list[float]]:
    """Returns mean IoU over non-ignored classes and per-class IoU list."""
    import math

    num_classes = cm.shape[0]
    ious: list[float] = []
    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp
        denom = tp + fp + fn
        ious.append(tp / denom if denom > 0 else float("nan"))
    valid = [x for x in ious if not math.isnan(x)]
    miou = sum(valid) / len(valid) if valid else 0.0
    return miou, ious
