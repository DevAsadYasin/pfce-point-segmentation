"""Remote-sensing style segmentation data with simulated point supervision."""

from __future__ import annotations

import random
from typing import Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset

# LoveDA / synthetic: 8 logits for mask values 0..7 where 0 = no-data (ignored for metrics at 0)
NUM_CLASSES = 8
IGNORE_INDEX = 0


def sample_random_points(mask: Tensor, num_points: int, generator: random.Random | None = None) -> Tensor:
    """Binary mask [H,W] with exactly `num_points` ones at random valid (mask!=0) pixels."""
    rng = generator or random
    h, w = mask.shape
    valid = (mask != 0).nonzero(as_tuple=False)  # [N,2]
    if valid.numel() == 0:
        return torch.zeros((h, w), dtype=torch.bool)
    n_valid = valid.size(0)
    k = min(num_points, n_valid)
    idx = torch.randperm(n_valid, device=mask.device)[:k]
    chosen = valid[idx]
    out = torch.zeros((h, w), dtype=torch.bool, device=mask.device)
    out[chosen[:, 0], chosen[:, 1]] = True
    return out


def sample_class_balanced_points(mask: Tensor, num_points: int, generator: random.Random | None = None) -> Tensor:
    """Spread points across classes present in crop (roughly equal per class, remainder random)."""
    rng = generator or random
    h, w = mask.shape
    device = mask.device
    classes = torch.unique(mask[mask != 0])
    if classes.numel() == 0:
        return torch.zeros((h, w), dtype=torch.bool, device=device)
    n_cls = int(classes.numel())
    base = num_points // n_cls
    remainder = num_points % n_cls
    chosen_list: list[Tensor] = []
    for c in classes.tolist():
        coords = (mask == int(c)).nonzero(as_tuple=False)
        if coords.numel() == 0:
            continue
        take = base + (1 if remainder > 0 else 0)
        remainder = max(0, remainder - 1)
        take = min(take, coords.size(0))
        perm = torch.randperm(coords.size(0), device=device)[:take]
        chosen_list.append(coords[perm])
    if not chosen_list:
        return torch.zeros((h, w), dtype=torch.bool, device=device)
    chosen = torch.cat(chosen_list, dim=0)
    if chosen.size(0) > num_points:
        perm = torch.randperm(chosen.size(0), device=device)[:num_points]
        chosen = chosen[perm]
    out = torch.zeros((h, w), dtype=torch.bool, device=device)
    out[chosen[:, 0], chosen[:, 1]] = True
    return out


def random_crop_pair(image: Tensor, mask: Tensor, crop_size: int, rng: random.Random) -> tuple[Tensor, Tensor]:
    """image [3,H,W], mask [H,W] long."""
    _, h, w = image.shape
    if h < crop_size or w < crop_size:
        raise ValueError(f"Image {h}x{w} smaller than crop {crop_size}")
    top = rng.randint(0, h - crop_size)
    left = rng.randint(0, w - crop_size)
    image = image[:, top : top + crop_size, left : left + crop_size]
    mask = mask[top : top + crop_size, left : left + crop_size]
    return image, mask


class SyntheticRSDataset(Dataset):
    """Small in-memory RS-like tiles for smoke tests without downloading LoveDA."""

    def __init__(
        self,
        length: int = 500,
        size: int = 256,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.length = length
        self.size = size
        self.seed = seed
        g = torch.Generator().manual_seed(seed)
        self._images = torch.randint(0, 255, (length, 3, size, size), generator=g, dtype=torch.float32) / 255.0
        self._masks = torch.randint(1, NUM_CLASSES, (length, size, size), generator=g, dtype=torch.long)
        # sprinkle no-data (0)
        noise = torch.rand(length, size, size, generator=g)
        self._masks[noise < 0.05] = 0

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "image": self._images[index],
            "mask": self._masks[index],
        }


class LoveDAPointDataset(Dataset):
    """Wraps TorchGeo LoveDA with sync crop + point simulation."""

    def __init__(
        self,
        base: Dataset,
        crop_size: int = 512,
        num_points: int = 50,
        sampling: Literal["random", "balanced"] = "random",
        seed: int = 0,
    ) -> None:
        self.base = base
        self.crop_size = crop_size
        self.num_points = num_points
        self.sampling = sampling
        self.seed = seed

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        import random as py_random

        rng = py_random.Random(self.seed + index)
        sample = self.base[index]
        image: Tensor = sample["image"].float() / 255.0 if sample["image"].max() > 1.5 else sample["image"].float()
        mask: Tensor = sample["mask"].long()
        _, h, w = image.shape
        if h >= self.crop_size and w >= self.crop_size:
            image, mask = random_crop_pair(image, mask, self.crop_size, rng)
        point_mask = (
            sample_class_balanced_points(mask, self.num_points, rng)
            if self.sampling == "balanced"
            else sample_random_points(mask, self.num_points, rng)
        )
        return {
            "image": image,
            "mask": mask,
            "point_mask": point_mask,
        }


def build_dataset(
    name: str,
    root: str,
    split: str,
    download: bool,
    crop_size: int,
    num_points: int,
    sampling: Literal["random", "balanced"],
    seed: int,
    max_samples: int | None,
    synthetic_length: int = 400,
) -> Dataset:
    """Return a Dataset yielding dicts with image, mask, point_mask."""
    if name == "synthetic":
        n = min(max_samples, synthetic_length) if max_samples is not None else synthetic_length
        base = SyntheticRSDataset(length=n, size=crop_size, seed=seed)

        class _SyntheticPoint(Dataset):
            def __init__(self) -> None:
                self.base = base

            def __len__(self) -> int:
                return len(self.base)

            def __getitem__(self, i: int) -> dict[str, Tensor]:
                import random as py_random

                rng = py_random.Random(seed + i)
                s = self.base[i]
                m = s["mask"]
                pm = (
                    sample_class_balanced_points(m, num_points, rng)
                    if sampling == "balanced"
                    else sample_random_points(m, num_points, rng)
                )
                return {"image": s["image"], "mask": m, "point_mask": pm}

        return _SyntheticPoint()
    from torchgeo.datasets import LoveDA

    scene = ["urban"] if max_samples and max_samples < 1200 else ["urban", "rural"]
    base_loveda = LoveDA(root=root, split=split, scene=scene, download=download)
    if max_samples is not None:
        n = min(max_samples, len(base_loveda))

        class _Sub(Dataset):
            def __len__(self) -> int:
                return n

            def __getitem__(self, i: int) -> dict[str, Tensor]:
                return base_loveda[i]

        base_loveda = _Sub()
    return LoveDAPointDataset(
        base_loveda,
        crop_size=crop_size,
        num_points=num_points,
        sampling=sampling,
        seed=seed,
    )


def imagenet_normalize(x: Tensor) -> Tensor:
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (x - mean) / std
