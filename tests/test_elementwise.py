"""Element-wise ops: mul, sub, div, clamp, abs — common in normalization and gating."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from conftest import run_pipeline, assert_artifacts_exist, assert_weight_sizes_match


class ScaleShift(nn.Module):
    """Learned per-channel scale + shift (FiLM / AdaIN style, sans conditioning)."""
    def __init__(self, channels: int = 8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.shift = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        return x * self.scale + self.shift


class ClampedOutput(nn.Module):
    """Conv → clamp to [0, 1] (common in image output heads)."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 8, 1)

    def forward(self, x):
        return self.conv(x).clamp(0.0, 1.0)


class DiffMap(nn.Module):
    """Subtract two feature maps — used in residual error maps and flow models."""
    def __init__(self):
        super().__init__()
        self.conv_a = nn.Conv2d(8, 8, 3, padding=1)
        self.conv_b = nn.Conv2d(8, 8, 3, padding=1)

    def forward(self, x):
        return self.conv_a(x) - self.conv_b(x)


class NormalizedFeatures(nn.Module):
    """Manual L2 norm: x / (||x|| + eps) — appears in metric learning heads."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 16)

    def forward(self, x):
        x = self.fc(x)
        norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return x / norm


class AbsActivation(nn.Module):
    """abs used as an activation in some signal-processing networks."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 8, 3, padding=1)

    def forward(self, x):
        return torch.abs(self.conv(x))


@pytest.mark.parametrize("ModelClass,x_shape", [
    (ScaleShift,           (1, 8, 16, 16)),
    (ClampedOutput,        (1, 8, 16, 16)),
    (DiffMap,              (1, 8, 16, 16)),
    (NormalizedFeatures,   (1, 16)),
    (AbsActivation,        (1, 8, 16, 16)),
])
def test_elementwise_op(ModelClass, x_shape, tmp_path):
    result = run_pipeline(ModelClass(), (torch.randn(*x_shape),), tmp_path / ModelClass.__name__)
    assert_artifacts_exist(result["out_dir"])
    assert_weight_sizes_match(result["weight_entries"], result["blob"])


def test_scale_shift_params_in_weights(tmp_path):
    result = run_pipeline(ScaleShift(channels=8), (torch.randn(1, 8, 16, 16),), tmp_path)
    names = {w["name"] for w in result["weight_entries"]}
    assert any("scale" in n for n in names)
    assert any("shift" in n for n in names)
