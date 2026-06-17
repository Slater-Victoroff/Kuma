"""
ConvNeXt-style block — the closest thing to a real Niko/Nika building block.

ConvNeXt block structure:
    depthwise 7×7 → permute → LayerNorm → Linear (expand 4×) → GELU → Linear (contract) → permute → scale → residual
"""

from __future__ import annotations

import torch
import torch.nn as nn

from conftest import run_pipeline, assert_artifacts_exist, assert_weight_sizes_match, assert_ops_present


class ConvNeXtBlock(nn.Module):
    """
    Faithful reproduction of the ConvNeXt V1 block.
    Uses 1×1 convs for the channel MLP (equivalent to Linear after permute,
    but avoids the permute in this variant for export simplicity).
    """
    def __init__(self, dim: int = 32, expansion: int = 4) -> None:
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(dim * expansion, dim)
        self.gamma = nn.Parameter(torch.ones(dim) * 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dw_conv(x)
        # (B, C, H, W) → (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = x * self.gamma
        # (B, H, W, C) → (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        return x + shortcut


class ConvNeXtStage(nn.Module):
    """Two stacked ConvNeXt blocks — tests that stacked blocks export cleanly."""
    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.blocks = nn.Sequential(ConvNeXtBlock(dim), ConvNeXtBlock(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class ConvNeXtWithDownsample(nn.Module):
    """Stage → 2× spatial downsample via strided LayerNorm+Conv stem."""
    def __init__(self) -> None:
        super().__init__()
        self.stage = ConvNeXtBlock(dim=32)
        self.downsample = nn.Sequential(
            nn.LayerNorm([32, 8, 8]),  # channel-first LN before downsample
            nn.Conv2d(32, 64, kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(self.stage(x))


def test_convnext_block_basic(tmp_path):
    result = run_pipeline(ConvNeXtBlock(dim=32), (torch.randn(1, 32, 8, 8),), tmp_path)
    assert_artifacts_exist(result["out_dir"])
    assert_weight_sizes_match(result["weight_entries"], result["blob"])
    assert_ops_present(result["manifest"], "aten.convolution.default", "aten.add.Tensor")


def test_convnext_block_weight_count(tmp_path):
    model = ConvNeXtBlock(dim=32)
    result = run_pipeline(model, (torch.randn(1, 32, 8, 8),), tmp_path)
    # dw_conv.weight, dw_conv.bias, norm.weight, norm.bias,
    # pw1.weight, pw1.bias, pw2.weight, pw2.bias, gamma → 9 tensors
    assert len(result["weight_entries"]) == 9, (
        f"Expected 9 weight tensors, got {len(result['weight_entries'])}: "
        f"{[w['name'] for w in result['weight_entries']]}"
    )


def test_convnext_stage_two_blocks(tmp_path):
    result = run_pipeline(ConvNeXtStage(dim=32), (torch.randn(1, 32, 8, 8),), tmp_path)
    assert_artifacts_exist(result["out_dir"])
    assert_weight_sizes_match(result["weight_entries"], result["blob"])
    assert len(result["weight_entries"]) == 18  # 9 × 2 blocks


def test_convnext_with_downsample(tmp_path):
    result = run_pipeline(ConvNeXtWithDownsample(), (torch.randn(1, 32, 8, 8),), tmp_path)
    assert_artifacts_exist(result["out_dir"])
    assert_weight_sizes_match(result["weight_entries"], result["blob"])
    # output should be (1, 64, 4, 4)
    outputs = result["manifest"]["outputs"]
    assert len(outputs) == 1
    assert outputs[0].get("shape") == [1, 64, 4, 4]


def test_convnext_gamma_in_weights(tmp_path):
    result = run_pipeline(ConvNeXtBlock(dim=32), (torch.randn(1, 32, 8, 8),), tmp_path)
    names = {w["name"] for w in result["weight_entries"]}
    assert any("gamma" in n for n in names), f"gamma not found in weights: {sorted(names)}"
