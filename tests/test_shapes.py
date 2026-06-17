"""Shape manipulation: flatten, reshape, permute, transpose, squeeze/unsqueeze."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from conftest import run_pipeline, assert_artifacts_exist, assert_weight_sizes_match


class FlattenHead(nn.Module):
    """Conv → global pool → flatten → linear. Tests flatten in a real context."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 8)

    def forward(self, x):
        return self.fc(self.pool(self.conv(x)).flatten(1))


class PermuteNorm(nn.Module):
    """Permute BCHW → BHWC → LayerNorm → permute back. Standard ConvNeXt pattern."""
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(16)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class TransposeLinear(nn.Module):
    """Linear applied on last dim after a transpose — common in sequence models."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 8)

    def forward(self, x):
        # x: (B, 16, T) → transpose → (B, T, 16) → linear → (B, T, 8)
        return self.fc(x.transpose(1, 2))


class UnsqueezeSqueeze(nn.Module):
    """Unsqueeze to add a dim, run a conv, squeeze it back out."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x):
        # x: (B, C) → (B, 1, 1, C) → conv → (B, 1, 1, C) → (B, C)
        x = x.unsqueeze(1).unsqueeze(1)
        x = self.conv(x)
        return x.squeeze(1).squeeze(1)


class ReshapeEmbed(nn.Module):
    """Flatten spatial dims and run a linear projection (ViT patch embed style)."""
    def __init__(self):
        super().__init__()
        # 4×4 patch from 8-channel input → 16-dim embedding
        self.proj = nn.Linear(8 * 4 * 4, 16)

    def forward(self, x):
        # x: (B, 8, H, W); extract non-overlapping 4×4 patches via unfold
        B, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        # simple: just reshape to (B, C*H*W) then project a fixed-size input
        return self.proj(x.flatten(1))


@pytest.mark.parametrize("ModelClass,x_shape", [
    (FlattenHead,     (1, 8, 8, 8)),
    (PermuteNorm,     (1, 16, 8, 8)),
    (TransposeLinear, (1, 16, 32)),
    (UnsqueezeSqueeze,(1, 16)),
])
def test_shape_op(ModelClass, x_shape, tmp_path):
    result = run_pipeline(ModelClass(), (torch.randn(*x_shape),), tmp_path / ModelClass.__name__)
    assert_artifacts_exist(result["out_dir"])
    assert_weight_sizes_match(result["weight_entries"], result["blob"])


def test_reshape_embed_fixed_input(tmp_path):
    # Input must be exactly (1, 8, 4, 4) so flatten(1) → 128 matches Linear(128, 16)
    result = run_pipeline(ReshapeEmbed(), (torch.randn(1, 8, 4, 4),), tmp_path)
    assert_artifacts_exist(result["out_dir"])
    assert_weight_sizes_match(result["weight_entries"], result["blob"])
    assert result["manifest"]["outputs"][0]["shape"] == [1, 16]
