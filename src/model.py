"""Segmentation model with ImageNet-pretrained encoder."""

from __future__ import annotations

import torch.nn as nn
import segmentation_models_pytorch as smp


def build_model(num_classes: int = 8, encoder: str = "resnet34") -> nn.Module:
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    )
