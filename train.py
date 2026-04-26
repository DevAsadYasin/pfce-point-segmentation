#!/usr/bin/env python3
"""Train segmentation with partial CE / partial focal CE on simulated point labels."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import csv
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import NUM_CLASSES, build_dataset, imagenet_normalize
from src.losses import partial_cross_entropy, partial_focal_loss
from src.metrics import confusion_matrix, miou_from_confusion
from src.model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "point_mask": torch.stack([b["point_mask"] for b in batch]),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, device=device)
    for batch in loader:
        x = imagenet_normalize(batch["image"].to(device))
        y = batch["mask"].to(device)
        logits = model(x)
        cm += confusion_matrix(logits, y, NUM_CLASSES, ignore_index=0)
    miou, _ = miou_from_confusion(cm.cpu(), ignore_index=0)
    return float(miou)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    loss_name: str,
    focal_gamma: float,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        x = imagenet_normalize(batch["image"].to(device))
        mask = batch["mask"].to(device)
        pm = batch["point_mask"].to(device)
        logits = model(x)
        if loss_name == "focal":
            loss = partial_focal_loss(logits, mask, pm, gamma=focal_gamma)
        else:
            loss = partial_cross_entropy(logits, mask, pm)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Point-supervised RS segmentation (LoveDA or synthetic)")
    p.add_argument("--dataset", choices=("synthetic", "loveda"), default="synthetic")
    p.add_argument("--data-root", type=str, default="./data/loveda")
    p.add_argument("--download", action="store_true", help="Download LoveDA (large) via TorchGeo")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-points", type=int, default=50)
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--loss", choices=("focal", "ce"), default="focal", help="partial focal vs partial CE")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--sampling", choices=("random", "balanced"), default="random")
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--save-log", type=str, default=None, help="Append CSV row with run metrics")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = build_dataset(
        args.dataset,
        root=args.data_root,
        split="train",
        download=args.download,
        crop_size=args.crop_size,
        num_points=args.num_points,
        sampling=args.sampling,
        seed=args.seed,
        max_samples=args.max_train_samples,
    )
    val_ds = build_dataset(
        args.dataset,
        root=args.data_root,
        split="val",
        download=args.download,
        crop_size=args.crop_size,
        num_points=args.num_points,
        sampling=args.sampling,
        seed=args.seed + 1,
        max_samples=args.max_val_samples,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )

    model = build_model(num_classes=NUM_CLASSES).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_miou = 0.0
    last_miou = 0.0
    for epoch in range(args.epochs):
        loss_avg = train_one_epoch(model, train_loader, device, opt, args.loss, args.focal_gamma)
        last_miou = evaluate(model, val_loader, device)
        best_miou = max(best_miou, last_miou)
        print(f"epoch {epoch + 1}/{args.epochs}  train_loss={loss_avg:.4f}  val_mIoU={last_miou:.4f}")

    print(f"best_val_mIoU={best_miou:.4f}")

    if args.save_log:
        path = Path(args.save_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(
                    [
                        "dataset",
                        "num_points",
                        "sampling",
                        "loss",
                        "epochs",
                        "val_mIoU",
                        "best_val_mIoU",
                    ]
                )
            w.writerow(
                [
                    args.dataset,
                    args.num_points,
                    args.sampling,
                    args.loss,
                    args.epochs,
                    f"{last_miou:.6f}",
                    f"{best_miou:.6f}",
                ]
            )


if __name__ == "__main__":
    main()
