"""Tests for Conv2d variants — the most common ops in Niko/Nika-style models."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from conftest import run_pipeline, assert_artifacts_exist, assert_weight_sizes_match, assert_ops_present


class Conv2dBasic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        return self.conv(x)


class Conv2dNoBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        return self.conv(x)


class Conv2dDepthwise(nn.Module):
    """Depthwise conv as used in MobileNet/ConvNeXt."""
    def __init__(self):
        super().__init__()
        self.dw = nn.Conv2d(16, 16, kernel_size=7, padding=3, groups=16, bias=True)

    def forward(self, x):
        return self.dw(x)


class Conv1x1(nn.Module):
    """Pointwise conv — the other half of depthwise-separable."""
    def __init__(self):
        super().__init__()
        self.pw = nn.Conv2d(16, 32, kernel_size=1, bias=True)

    def forward(self, x):
        return self.pw(x)


class ConvStack(nn.Module):
    """Stacked conv → BN → ReLU block (backbone staple)."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ConvTranspose2dBasic(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)

    def forward(self, x):
        return self.upsample(x)


@pytest.mark.parametrize("ModelClass,x_shape", [
    (Conv2dBasic,       (1, 8, 16, 16)),
    (Conv2dNoBias,      (1, 8, 16, 16)),
    (Conv2dDepthwise,   (1, 16, 16, 16)),
    (Conv1x1,           (1, 16, 16, 16)),
    (ConvStack,         (1, 3, 16, 16)),
    (ConvTranspose2dBasic, (1, 16, 8, 8)),
])
def test_conv_variant(ModelClass, x_shape, tmp_path):
    result = run_pipeline(ModelClass(), (torch.randn(*x_shape),), tmp_path / ModelClass.__name__)
    assert_artifacts_exist(result["out_dir"])
    assert_weight_sizes_match(result["weight_entries"], result["blob"])
    assert_ops_present(result["manifest"], "aten.convolution.default")
    assert len(result["weight_entries"]) > 0


def test_conv_weight_count_no_bias(tmp_path):
    result = run_pipeline(Conv2dNoBias(), (torch.randn(1, 8, 16, 16),), tmp_path)
    names = {w["name"] for w in result["weight_entries"]}
    assert any("weight" in n for n in names)
    assert not any("bias" in n for n in names)


def test_conv_weight_shape(tmp_path):
    result = run_pipeline(Conv2dBasic(), (torch.randn(1, 8, 16, 16),), tmp_path)
    weight = next(w for w in result["weight_entries"] if "weight" in w["name"])
    assert weight["shape"] == [16, 8, 3, 3]
