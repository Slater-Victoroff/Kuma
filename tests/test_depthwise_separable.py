"""Depthwise-separable conv and MobileNet-style blocks — extremely common in efficient models."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from conftest import run_pipeline, assert_artifacts_exist, assert_weight_sizes_match, assert_ops_present


class DepthwiseSeparable(nn.Module):
    """Depthwise conv + pointwise conv — the MobileNet building block."""
    def __init__(self, channels: int = 16):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class InvertedResidual(nn.Module):
    """MobileNetV2 inverted residual: expand → dw → project, with residual if same shape."""
    def __init__(self, channels: int = 16, expansion: int = 4):
        super().__init__()
        mid = channels * expansion
        self.expand = nn.Conv2d(channels, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.dw = nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.project = nn.Conv2d(mid, channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU6()

    def forward(self, x):
        out = self.act(self.bn1(self.expand(x)))
        out = self.act(self.bn2(self.dw(out)))
        out = self.bn3(self.project(out))
        return out + x


class DSConvStack(nn.Module):
    """Three stacked depthwise-separable blocks — tests repeated DW patterns."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            DepthwiseSeparable(16),
            DepthwiseSeparable(16),
            DepthwiseSeparable(16),
        )

    def forward(self, x):
        return self.layers(x)


@pytest.mark.parametrize("ModelClass,x_shape", [
    (DepthwiseSeparable,  (1, 16, 16, 16)),
    (InvertedResidual,    (1, 16, 16, 16)),
    (DSConvStack,         (1, 16, 16, 16)),
])
def test_dw_sep_variant(ModelClass, x_shape, tmp_path):
    result = run_pipeline(ModelClass(), (torch.randn(*x_shape),), tmp_path / ModelClass.__name__)
    assert_artifacts_exist(result["out_dir"])
    assert_weight_sizes_match(result["weight_entries"], result["blob"])
    assert_ops_present(result["manifest"], "aten.convolution.default")


def test_dw_groups_reflected_in_weight_shape(tmp_path):
    result = run_pipeline(DepthwiseSeparable(channels=16), (torch.randn(1, 16, 16, 16),), tmp_path)
    dw_weight = next(w for w in result["weight_entries"] if "dw" in w["name"] and "weight" in w["name"])
    # groups=channels → weight shape is (channels, 1, kH, kW)
    assert dw_weight["shape"] == [16, 1, 3, 3]
