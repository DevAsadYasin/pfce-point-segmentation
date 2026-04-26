#!/usr/bin/env python3
"""Run a small experiment grid and append results to experiments/logs/results.csv."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    log = root / "experiments" / "logs" / "results.csv"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="synthetic", choices=("synthetic", "loveda"))
    parser.add_argument("--data-root", default=str(root / "data" / "loveda"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int, default=200)
    parser.add_argument("--max-val-samples", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Experiment A: point density x sampling (partial focal / pfCE)
    grid_a: list[tuple[int, str]] = [(n, s) for n in (10, 50, 200) for s in ("random", "balanced")]
    # Experiment B: partial CE only (compare to pfCE row for num_points=50, sampling=random from grid A)
    grid_b_loss = ("ce",)

    exe = sys.executable
    train = root / "train.py"
    for num_points, sampling in grid_a:
        cmd = [
            exe,
            str(train),
            "--dataset",
            args.dataset,
            "--data-root",
            args.data_root,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--crop-size",
            str(args.crop_size),
            "--num-points",
            str(num_points),
            "--sampling",
            sampling,
            "--loss",
            "focal",
            "--max-train-samples",
            str(args.max_train_samples),
            "--max-val-samples",
            str(args.max_val_samples),
            "--seed",
            str(args.seed),
            "--save-log",
            str(log),
        ]
        if args.download:
            cmd.append("--download")
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, cwd=str(root), check=True)

    n0, s0 = 50, "random"
    for loss_name in grid_b_loss:
        cmd = [
            exe,
            str(train),
            "--dataset",
            args.dataset,
            "--data-root",
            args.data_root,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--crop-size",
            str(args.crop_size),
            "--num-points",
            str(n0),
            "--sampling",
            s0,
            "--loss",
            loss_name,
            "--max-train-samples",
            str(args.max_train_samples),
            "--max-val-samples",
            str(args.max_val_samples),
            "--seed",
            str(args.seed),
            "--save-log",
            str(log),
        ]
        if args.download:
            cmd.append("--download")
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, cwd=str(root), check=True)

    print(f"Wrote aggregated runs to {log}")


if __name__ == "__main__":
    main()
