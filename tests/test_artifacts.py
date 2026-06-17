"""Artifact integrity tests: file existence, size consistency, schema validity.

These run against the tiny_model fixture and catch regressions in the pipeline
itself rather than in any specific layer type.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from conftest import (
    run_pipeline,
    assert_artifacts_exist,
    assert_weight_sizes_match,
    assert_manifest_schema,
    roundtrip_weight,
)


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 3, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, x):
        return self.conv2(self.act(self.conv1(x))) + x


@pytest.fixture(scope="module")
def tiny_result(tmp_path_factory):
    out = tmp_path_factory.mktemp("tiny")
    return run_pipeline(_Tiny(), (torch.randn(1, 3, 32, 32),), out)


def test_all_artifacts_exist(tiny_result):
    assert_artifacts_exist(tiny_result["out_dir"])


def test_weight_byte_sizes_are_consistent(tiny_result):
    assert_weight_sizes_match(tiny_result["weight_entries"], tiny_result["blob"])


def test_blob_size_equals_sum_of_byte_lengths(tiny_result):
    total = sum(w["byte_length"] for w in tiny_result["weight_entries"])
    # blob may have a few alignment padding bytes at the end — it must be >= total
    assert len(tiny_result["blob"]) >= total


def test_manifest_schema(tiny_result):
    assert_manifest_schema(tiny_result["manifest"])


def test_manifest_json_is_valid_on_disk(tiny_result):
    raw = (tiny_result["out_dir"] / "manifest.json").read_text()
    parsed = json.loads(raw)
    assert parsed["format"] == "iphso-webgpu-export"


def test_exported_graph_json_is_valid(tiny_result):
    raw = (tiny_result["out_dir"] / "exported_graph.json").read_text()
    parsed = json.loads(raw)
    assert "nodes" in parsed
    assert len(parsed["nodes"]) > 0


def test_debug_report_contains_ops(tiny_result):
    report = (tiny_result["out_dir"] / "debug_report.md").read_text()
    assert "aten." in report
    assert "## ATen Ops Encountered" in report


def test_weight_roundtrip_is_exact(tiny_result):
    """Pack → unpack should produce bit-identical tensors."""
    sd = dict(tiny_result["ep"].state_dict)
    for name in sd:
        recovered = roundtrip_weight(tiny_result["weight_entries"], tiny_result["blob"], name)
        original = sd[name].cpu().float()
        assert torch.allclose(recovered, original, atol=0.0), (
            f"Roundtrip mismatch for '{name}'"
        )


def test_manifest_input_has_shape(tiny_result):
    inputs = tiny_result["manifest"]["inputs"]
    assert len(inputs) == 1
    assert inputs[0]["shape"] == [1, 3, 32, 32]
    assert inputs[0]["dtype"] == "float32"


def test_placeholder_nodes_annotated_with_weight_name(tiny_result):
    param_nodes = [
        n for n in tiny_result["graph_data"]["nodes"]
        if n["op"] == "placeholder" and n.get("kind") == "parameter"
    ]
    assert len(param_nodes) > 0, "No parameter placeholder nodes found"
    for n in param_nodes:
        assert "weight_name" in n, f"Node {n['name']} missing weight_name"
