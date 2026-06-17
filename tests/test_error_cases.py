"""Tests that the exporter fails loudly on inputs it shouldn't accept."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from iphso_webgpu_export.export import run_export, _validate_inputs, _validate_parameters


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


def test_non_tuple_input_rejected(tmp_path):
    with pytest.raises((TypeError, Exception)):
        # Pass a list instead of tuple — should raise
        from iphso_webgpu_export.export import _load_factory
        import sys, types
        # Inject a synthetic module with a bad factory
        mod = types.ModuleType("_bad_input_test")
        mod.bad_factory = lambda: [torch.randn(1, 3, 4, 4)]  # list, not tuple
        sys.modules["_bad_input_test"] = mod
        try:
            run_export("_bad_input_test:bad_factory.__class__", "_bad_input_test:bad_factory", tmp_path)
        finally:
            del sys.modules["_bad_input_test"]
