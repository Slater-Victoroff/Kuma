"""Shared fixtures for the iphso-webgpu-export test suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from iphso_webgpu_export.graph import write_graph
from iphso_webgpu_export.pack_weights import write_weights
from iphso_webgpu_export.manifest import write_manifest
from iphso_webgpu_export.debug import write_debug_report


def run_pipeline(
    model: nn.Module,
    example_inputs: tuple,
    out_dir: Path,
) -> dict[str, Any]:
    """
    Run the full export pipeline against a model and return a dict of all outputs.
    This is the canonical test harness — tests should call this, not reimport internals.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        ep = torch.export.export(model, example_inputs)

    graph_data = write_graph(ep, out_dir)
    blob, weight_entries, skipped = write_weights(ep, out_dir)
    warnings = [f"skipped non-float32 tensor: {s}" for s in skipped]
    manifest = write_manifest(ep, weight_entries, graph_data, warnings, out_dir)
    write_debug_report(ep, graph_data, weight_entries, manifest, out_dir)

    return {
        "ep": ep,
        "graph_data": graph_data,
        "blob": blob,
        "weight_entries": weight_entries,
        "skipped": skipped,
        "manifest": manifest,
        "out_dir": out_dir,
    }


# ── Common assertion helpers ─────────────────────────────────────────────────

def assert_artifacts_exist(out_dir: Path) -> None:
    for fname in ["exported_graph.json", "weights.f32.bin", "manifest.json", "debug_report.md"]:
        assert (out_dir / fname).exists(), f"Missing artifact: {fname}"


def assert_weight_sizes_match(weight_entries: list[dict], blob: bytes) -> None:
    """weights.f32.bin size and every byte_offset+byte_length must be consistent."""
    for w in weight_entries:
        end = w["byte_offset"] + w["byte_length"]
        assert end <= len(blob), (
            f"Weight '{w['name']}' claims offset {w['byte_offset']} + "
            f"length {w['byte_length']} = {end} but blob is only {len(blob)} bytes"
        )
        expected_bytes = w["n_elements"] * 4  # float32
        assert w["byte_length"] == expected_bytes, (
            f"Weight '{w['name']}': byte_length {w['byte_length']} != "
            f"n_elements {w['n_elements']} * 4 = {expected_bytes}"
        )


def assert_manifest_schema(manifest: dict) -> None:
    for key in ("format", "format_version", "inputs", "outputs", "weights", "graph", "warnings"):
        assert key in manifest, f"manifest.json missing key: {key!r}"
    assert manifest["format"] == "iphso-webgpu-export"
    assert manifest["format_version"] == 0
    assert isinstance(manifest["weights"], list)
    assert "nodes" in manifest["graph"]
    assert "op_counts" in manifest["graph"]


def assert_ops_present(manifest: dict, *expected_ops: str) -> None:
    found = set(manifest["graph"]["op_counts"].keys())
    for op in expected_ops:
        assert op in found, f"Expected op '{op}' not found in graph. Got: {sorted(found)}"


def roundtrip_weight(weight_entries: list[dict], blob: bytes, name: str) -> torch.Tensor:
    """Read a named weight back out of the blob and return it as a float32 tensor."""
    entry = next(w for w in weight_entries if w["name"] == name)
    raw = blob[entry["byte_offset"] : entry["byte_offset"] + entry["byte_length"]]
    import numpy as np
    arr = np.frombuffer(raw, dtype="<f4").copy()
    return torch.from_numpy(arr).reshape(entry["shape"])
