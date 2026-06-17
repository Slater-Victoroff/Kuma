"""Multi-input models — two or three tensors as separate inputs to the graph."""

from __future__ import annotations

import torch
import torch.nn as nn

from conftest import run_pipeline, assert_artifacts_exist, assert_weight_sizes_match


class TwoStreamFuse(nn.Module):
    """Two independent input tensors fused by addition after separate convs."""
    def __init__(self):
        super().__init__()
        self.branch_a = nn.Conv2d(8, 8, 3, padding=1)
        self.branch_b = nn.Conv2d(8, 8, 3, padding=1)

    def forward(self, x, y):
        return self.branch_a(x) + self.branch_b(y)


class ConditionedConv(nn.Module):
    """
    Feature map x conditioned by a vector c (FiLM-style without the MLP).
    c is a second input that provides per-channel scale.
    """
    def __init__(self, channels: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x, c):
        # c: (B, channels) → scale each channel
        return self.conv(x) * c[:, :, None, None]


class ResidualWithMask(nn.Module):
    """Feature map + binary mask as separate inputs (common in inpainting models)."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8 + 1, 8, 3, padding=1)  # concat mask on channel dim

    def forward(self, x, mask):
        return self.conv(torch.cat([x, mask], dim=1))


def test_two_stream_fuse(tmp_path):
    x = torch.randn(1, 8, 16, 16)
    y = torch.randn(1, 8, 16, 16)
    result = run_pipeline(TwoStreamFuse(), (x, y), tmp_path)
    assert_artifacts_exist(result["out_dir"])
    assert_weight_sizes_match(result["weight_entries"], result["blob"])
    inputs = result["manifest"]["inputs"]
    assert len(inputs) == 2
    assert inputs[0]["shape"] == [1, 8, 16, 16]
    assert inputs[1]["shape"] == [1, 8, 16, 16]


def test_conditioned_conv(tmp_path):
    x = torch.randn(1, 8, 16, 16)
    c = torch.randn(1, 8)
    result = run_pipeline(ConditionedConv(channels=8), (x, c), tmp_path)
    assert_artifacts_exist(result["out_dir"])
    assert_weight_sizes_match(result["weight_entries"], result["blob"])
    assert len(result["manifest"]["inputs"]) == 2


def test_residual_with_mask(tmp_path):
    x = torch.randn(1, 8, 16, 16)
    mask = torch.zeros(1, 1, 16, 16)  # binary mask, still float32
    result = run_pipeline(ResidualWithMask(), (x, mask), tmp_path)
    assert_artifacts_exist(result["out_dir"])
    assert_weight_sizes_match(result["weight_entries"], result["blob"])
    assert result["manifest"]["outputs"][0]["shape"] == [1, 8, 16, 16]
