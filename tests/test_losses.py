"""Sanity checks: masked full-image CE matches partial CE when mask is all ones."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import torch

from src.losses import partial_cross_entropy, partial_focal_loss


def test_partial_ce_matches_full_when_all_labeled() -> None:
    torch.manual_seed(0)
    b, c, h, w = 2, 4, 8, 8
    logits = torch.randn(b, c, h, w)
    target = torch.randint(0, c, (b, h, w))
    mask = torch.ones(b, h, w, dtype=torch.bool)
    p = partial_cross_entropy(logits, target, mask)
    full = torch.nn.functional.cross_entropy(logits, target, reduction="mean")
    assert torch.allclose(p, full, rtol=1e-5, atol=1e-5)


def test_partial_focal_finite() -> None:
    torch.manual_seed(1)
    b, c, h, w = 1, 3, 16, 16
    logits = torch.randn(b, c, h, w)
    target = torch.randint(1, c, (b, h, w))
    mask = torch.zeros(b, h, w, dtype=torch.bool)
    mask[:, 2:10, 2:10] = True
    loss = partial_focal_loss(logits, target, mask, gamma=2.0)
    assert loss.ndim == 0 and torch.isfinite(loss)


if __name__ == "__main__":
    test_partial_ce_matches_full_when_all_labeled()
    test_partial_focal_finite()
    print("ok")
