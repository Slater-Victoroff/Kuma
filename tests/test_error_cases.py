"""Tests that the exporter fails loudly on inputs it shouldn't accept."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from kuma.export import export_program, _validate_inputs, _validate_parameters


class _TinyFloat16(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 1).half()

    def forward(self, x):
        return self.conv(x)


def test_float16_input_rejected():
    with pytest.raises(ValueError, match="float16|dtype"):
        _validate_inputs((torch.randn(1, 4, 4, 4).half(),))


def test_float16_parameter_rejected():
    with pytest.raises(ValueError, match="float16|dtype"):
        _validate_parameters(_TinyFloat16())


def test_non_cpu_input_rejected():
    # Only attempt if CUDA is available; otherwise skip gracefully.
    pytest.importorskip("torch.cuda")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    with pytest.raises(ValueError, match="device"):
        _validate_inputs((torch.randn(1, 4, 4, 4).cuda(),))


def test_non_tuple_input_rejected():
    model = nn.Conv2d(3, 3, 1)
    with pytest.raises(TypeError, match="tuple"):
        export_program(model, [torch.randn(1, 3, 4, 4)])  # list, not tuple
