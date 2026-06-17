"""Tests for activation functions that appear in the target model families."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from conftest import run_pipeline, assert_artifacts_exist, assert_weight_sizes_match


def _wrap(act: nn.Module) -> nn.Module:
    class _Wrapped(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(8, 8, 1)
            self.act = act

        def forward(self, x):
            return self.act(self.conv(x))

    return _Wrapped()


@pytest.mark.parametrize("act,name", [
    (nn.GELU(),       "gelu"),
    (nn.ReLU(),       "relu"),
    (nn.SiLU(),       "silu"),
    (nn.Hardswish(),  "hardswish"),
    (nn.Sigmoid(),    "sigmoid"),
    (nn.Tanh(),       "tanh"),
])
def test_activation_exports(act, name, tmp_path):
    model = _wrap(act)
    result = run_pipeline(model, (torch.randn(1, 8, 8, 8),), tmp_path / name)
    assert_artifacts_exist(result["out_dir"])
    assert_weight_sizes_match(result["weight_entries"], result["blob"])
    # Each activation must produce at least one call_function node
    call_nodes = [n for n in result["graph_data"]["nodes"] if n["op"] == "call_function"]
    assert len(call_nodes) > 0
