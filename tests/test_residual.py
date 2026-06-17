"""Tests for residual / skip connections with and without projection."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from conftest import run_pipeline, assert_artifacts_exist, assert_weight_sizes_match, assert_ops_present


class PlainResidual(nn.Module):
    """Skip add with no channel change — no projection needed."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 16, 3, padding=1)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.conv(x)) + x


class ProjectedResidual(nn.Module):
    """Skip add where in_channels != out_channels — needs a 1×1 projection."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 16, 3, padding=1)
        self.act = nn.ReLU()
        self.proj = nn.Conv2d(8, 16, 1)

    def forward(self, x):
        return self.act(self.conv(x)) + self.proj(x)


class BottleneckBlock(nn.Module):
    """ResNet-style bottleneck: 1×1 → 3×3 → 1×1 with projection shortcut."""
    def __init__(self, inplanes: int = 16, planes: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, inplanes, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(inplanes)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.act(out + x)


@pytest.mark.parametrize("ModelClass,x_shape", [
    (PlainResidual,      (1, 16, 8, 8)),
    (ProjectedResidual,  (1, 8, 8, 8)),
    (BottleneckBlock,    (1, 16, 8, 8)),
])
def test_residual_variant(ModelClass, x_shape, tmp_path):
    result = run_pipeline(ModelClass(), (torch.randn(*x_shape),), tmp_path / ModelClass.__name__)
    assert_artifacts_exist(result["out_dir"])
    assert_weight_sizes_match(result["weight_entries"], result["blob"])
    assert_ops_present(result["manifest"], "aten.add.Tensor")
