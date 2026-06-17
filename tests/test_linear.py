"""Tests for Linear / MLP blocks."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from conftest import run_pipeline, assert_artifacts_exist, assert_weight_sizes_match, assert_ops_present


class LinearBasic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 32)

    def forward(self, x):
        return self.fc(x)


class MLP(nn.Module):
    """Two-layer MLP with GELU — common in transformer FFN blocks."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 256)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(256, 64)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class ConvNeXtFFN(nn.Module):
    """ConvNeXt-style channel-MLP via 1×1 convs (equivalent to Linear on spatial features)."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Conv2d(32, 128, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(128, 32, 1)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


@pytest.mark.parametrize("ModelClass,x_shape", [
    (LinearBasic,    (1, 64)),
    (MLP,            (1, 64)),
    (ConvNeXtFFN,    (1, 32, 8, 8)),
])
def test_linear_variant(ModelClass, x_shape, tmp_path):
    result = run_pipeline(ModelClass(), (torch.randn(*x_shape),), tmp_path / ModelClass.__name__)
    assert_artifacts_exist(result["out_dir"])
    assert_weight_sizes_match(result["weight_entries"], result["blob"])


def test_mlp_has_two_weight_tensors(tmp_path):
    result = run_pipeline(MLP(), (torch.randn(1, 64),), tmp_path)
    weight_names = [w["name"] for w in result["weight_entries"] if "weight" in w["name"]]
    assert len(weight_names) == 2
