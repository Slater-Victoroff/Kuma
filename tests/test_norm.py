"""Tests for normalization layers: BatchNorm2d, LayerNorm, GroupNorm, InstanceNorm."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from conftest import run_pipeline, assert_artifacts_exist, assert_weight_sizes_match


class BNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 16, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)

    def forward(self, x):
        return self.bn(self.conv(x))


class LNModel(nn.Module):
    """LayerNorm on channel dim — common in ViT and ConvNeXt."""
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(32)
        self.fc = nn.Linear(32, 32)

    def forward(self, x):
        return self.fc(self.ln(x))


class GNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 16, 3, padding=1, bias=False)
        self.gn = nn.GroupNorm(4, 16)

    def forward(self, x):
        return self.gn(self.conv(x))


class ConvNeXtNorm(nn.Module):
    """
    ConvNeXt uses LayerNorm on (B, C, H, W) by treating C as the normed dim
    after permuting.  This tests the permute → LayerNorm → permute pattern.
    """
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(16)

    def forward(self, x):
        # x: (B, C, H, W) → (B, H, W, C) → LayerNorm → (B, C, H, W)
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        return x.permute(0, 3, 1, 2)


@pytest.mark.parametrize("ModelClass,x_shape", [
    (BNModel,       (1, 8, 16, 16)),
    (LNModel,       (4, 32)),
    (GNModel,       (1, 16, 16, 16)),
    (ConvNeXtNorm,  (1, 16, 8, 8)),
])
def test_norm_variant(ModelClass, x_shape, tmp_path):
    result = run_pipeline(ModelClass(), (torch.randn(*x_shape),), tmp_path / ModelClass.__name__)
    assert_artifacts_exist(result["out_dir"])
    assert_weight_sizes_match(result["weight_entries"], result["blob"])


def test_batchnorm_running_stats_packed(tmp_path):
    """running_mean and running_var (float32 buffers) must be in the weight blob.
    num_batches_tracked (int64) is training-only and should be skipped with a warning."""
    result = run_pipeline(BNModel(), (torch.randn(1, 8, 16, 16),), tmp_path)
    names = {w["name"] for w in result["weight_entries"]}
    assert any("running_mean" in n for n in names), f"running_mean not found in: {sorted(names)}"
    assert any("running_var" in n for n in names), f"running_var not found in: {sorted(names)}"
    # num_batches_tracked is int64 — it should be skipped, not packed
    assert not any("num_batches_tracked" in n for n in names)
    assert any("num_batches_tracked" in s for s in result["skipped"])
