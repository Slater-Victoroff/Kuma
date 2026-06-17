"""Pooling layers: MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, global-pool classifier head."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from conftest import run_pipeline, assert_artifacts_exist, assert_weight_sizes_match


class MaxPoolModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        return self.pool(self.conv(x))


class AvgPoolModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 8, 3, padding=1)
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        return self.pool(self.conv(x))


class AdaptiveAvgPoolModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

    def forward(self, x):
        return self.pool(self.conv(x))


class GlobalAvgPoolClassifier(nn.Module):
    """Conv → global avg pool → flatten → linear. Standard image classifier head."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.act = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(16, 10)

    def forward(self, x):
        x = self.act(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.head(x)


class MaxPoolStrided(nn.Module):
    """MaxPool used as a stride-2 spatial downsampler (common before dense stages)."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 7, padding=3, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv(x))))


@pytest.mark.parametrize("ModelClass,x_shape", [
    (MaxPoolModel,            (1, 8, 16, 16)),
    (AvgPoolModel,            (1, 8, 16, 16)),
    (AdaptiveAvgPoolModel,    (1, 8, 16, 16)),
    (GlobalAvgPoolClassifier, (1, 3, 32, 32)),
    (MaxPoolStrided,          (1, 3, 64, 64)),
])
def test_pool_variant(ModelClass, x_shape, tmp_path):
    result = run_pipeline(ModelClass(), (torch.randn(*x_shape),), tmp_path / ModelClass.__name__)
    assert_artifacts_exist(result["out_dir"])
    assert_weight_sizes_match(result["weight_entries"], result["blob"])


def test_classifier_head_output_shape(tmp_path):
    result = run_pipeline(GlobalAvgPoolClassifier(), (torch.randn(1, 3, 32, 32),), tmp_path)
    outputs = result["manifest"]["outputs"]
    assert outputs[0]["shape"] == [1, 10]
