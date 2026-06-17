"""torch.cat and skip-connection patterns: channel concat, U-Net fuse, multi-scale."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from conftest import run_pipeline, assert_artifacts_exist, assert_weight_sizes_match, assert_ops_present


class CatChannels(nn.Module):
    """Concatenate two feature maps along the channel dim."""
    def __init__(self):
        super().__init__()
        self.conv_a = nn.Conv2d(8, 8, 3, padding=1)
        self.conv_b = nn.Conv2d(8, 8, 3, padding=1)
        self.fuse = nn.Conv2d(16, 8, 1)

    def forward(self, x):
        a = self.conv_a(x)
        b = self.conv_b(x)
        return self.fuse(torch.cat([a, b], dim=1))


class UNetFuseBlock(nn.Module):
    """Upsample + cat with skip + conv — the core U-Net decoder fusion."""
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.fuse = nn.Conv2d(16 + 8, 8, 3, padding=1)
        self.act = nn.ReLU()

    def forward(self, low_res, skip):
        # low_res: (B, 16, H/2, W/2), skip: (B, 8, H, W)
        up = self.up(low_res)
        return self.act(self.fuse(torch.cat([up, skip], dim=1)))


class MultiScaleFuse(nn.Module):
    """Cat three feature maps from different scales (FPN-style)."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3 * 8, 8, 1)

    def forward(self, x):
        a = x
        b = torch.nn.functional.avg_pool2d(x, 2, 2)
        b = torch.nn.functional.interpolate(b, scale_factor=2, mode="nearest")
        c = torch.nn.functional.avg_pool2d(x, 4, 4)
        c = torch.nn.functional.interpolate(c, scale_factor=4, mode="nearest")
        return self.proj(torch.cat([a, b, c], dim=1))


def test_cat_channels(tmp_path):
    result = run_pipeline(CatChannels(), (torch.randn(1, 8, 16, 16),), tmp_path)
    assert_artifacts_exist(result["out_dir"])
    assert_weight_sizes_match(result["weight_entries"], result["blob"])
    assert_ops_present(result["manifest"], "aten.cat.default")


def test_unet_fuse_block(tmp_path):
    low = torch.randn(1, 16, 8, 8)
    skip = torch.randn(1, 8, 16, 16)
    result = run_pipeline(UNetFuseBlock(), (low, skip), tmp_path)
    assert_artifacts_exist(result["out_dir"])
    assert_weight_sizes_match(result["weight_entries"], result["blob"])
    assert_ops_present(result["manifest"], "aten.cat.default")
    assert result["manifest"]["inputs"][0]["shape"] == [1, 16, 8, 8]
    assert result["manifest"]["inputs"][1]["shape"] == [1, 8, 16, 16]


def test_multi_scale_fuse(tmp_path):
    result = run_pipeline(MultiScaleFuse(), (torch.randn(1, 8, 16, 16),), tmp_path)
    assert_artifacts_exist(result["out_dir"])
    assert_weight_sizes_match(result["weight_entries"], result["blob"])
