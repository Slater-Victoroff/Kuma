"""Upsampling: nn.Upsample (nearest/bilinear), ConvTranspose2d decoder blocks."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from conftest import run_pipeline, assert_artifacts_exist, assert_weight_sizes_match


class UpsampleNearest(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(8, 8, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class UpsampleBilinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(8, 8, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class DecoderBlock(nn.Module):
    """Upsample → conv → BN → ReLU, the staple of U-Net decoder stages."""
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(16, 8, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(8)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(self.up(x))))


class ConvTransposeDecoder(nn.Module):
    """ConvTranspose2d as a learnable upsampler (common in GANs and segmentation heads)."""
    def __init__(self):
        super().__init__()
        self.up = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.up(x))


class PixelShuffle(nn.Module):
    """Conv → PixelShuffle for sub-pixel upsampling (common in super-resolution)."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 8 * 4, 3, padding=1)  # 4 = upscale_factor^2
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        return self.shuffle(self.conv(x))


@pytest.mark.parametrize("ModelClass,x_shape", [
    (UpsampleNearest,       (1, 8, 8, 8)),
    (UpsampleBilinear,      (1, 8, 8, 8)),
    (DecoderBlock,          (1, 16, 8, 8)),
    (ConvTransposeDecoder,  (1, 16, 8, 8)),
    (PixelShuffle,          (1, 8, 8, 8)),
])
def test_upsample_variant(ModelClass, x_shape, tmp_path):
    result = run_pipeline(ModelClass(), (torch.randn(*x_shape),), tmp_path / ModelClass.__name__)
    assert_artifacts_exist(result["out_dir"])
    assert_weight_sizes_match(result["weight_entries"], result["blob"])


def test_upsample_doubles_spatial(tmp_path):
    result = run_pipeline(UpsampleNearest(), (torch.randn(1, 8, 8, 8),), tmp_path)
    outputs = result["manifest"]["outputs"]
    assert outputs[0]["shape"] == [1, 8, 16, 16]
