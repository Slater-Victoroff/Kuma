"""Core export: import model + inputs, run torch.export.export, return ExportedProgram."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable

import torch
import torch.export


def _load_factory(spec: str) -> Callable:
    """Load a callable from a 'dotted.module:attr' spec string."""
    if ":" not in spec:
        raise ValueError(f"Factory spec must be 'module:attr', got {spec!r}")
    module_path, attr = spec.rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _validate_inputs(example_inputs: tuple) -> None:
    for i, t in enumerate(example_inputs):
        if not isinstance(t, torch.Tensor):
            continue
        if t.dtype != torch.float32:
            raise ValueError(
                f"Input[{i}] has dtype {t.dtype}; only float32 is supported."
            )
        if t.device.type != "cpu":
            raise ValueError(
                f"Input[{i}] is on device '{t.device}'; only cpu is supported."
            )


def _validate_parameters(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        if param.dtype != torch.float32:
            raise ValueError(
                f"Parameter '{name}' has dtype {param.dtype}; only float32 is supported."
            )
    for name, buf in model.named_buffers():
        if buf.dtype not in (torch.float32, torch.int64, torch.bool):
            raise ValueError(
                f"Buffer '{name}' has dtype {buf.dtype}; only float32/int64/bool buffers are supported."
            )


def run_export(
    model_spec: str,
    input_spec: str,
    out_dir: Path,
) -> torch.export.ExportedProgram:
    """
    Load model and example inputs from factory specs, validate, and run torch.export.export.

    Fails loudly on unsupported dtypes, non-CPU tensors, or dynamic-shape graphs.
    """
    model_factory = _load_factory(model_spec)
    input_factory = _load_factory(input_spec)

    model = model_factory()
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Model factory must return nn.Module, got {type(model)}")
    model.eval()

    example_inputs = input_factory()
    if not isinstance(example_inputs, tuple):
        raise TypeError(
            f"Input factory must return a tuple of tensors, got {type(example_inputs)}"
        )

    _validate_inputs(example_inputs)
    _validate_parameters(model)

    with torch.no_grad():
        ep = torch.export.export(model, example_inputs)

    return ep
