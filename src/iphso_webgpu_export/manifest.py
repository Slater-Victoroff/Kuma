"""Build and write manifest.json — the self-contained bundle descriptor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.export

try:
    from torch.export.graph_signature import InputKind, OutputKind
except ImportError:
    from torch._export.graph_signature import InputKind, OutputKind  # type: ignore[no-redef]


_DTYPE_NAMES: dict[torch.dtype, str] = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float64: "float64",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.bool: "bool",
}


def _dtype_str(dtype: Any) -> str | None:
    if dtype is None:
        return None
    return _DTYPE_NAMES.get(dtype, str(dtype))


def _node_val_meta(node_name: str, graph_nodes: list[dict[str, Any]]) -> dict[str, Any]:
    for n in graph_nodes:
        if n["name"] == node_name:
            return n.get("meta", {})
    return {}


def build_manifest(
    ep: torch.export.ExportedProgram,
    weight_entries: list[dict[str, Any]],
    graph_data: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    sig = ep.graph_signature
    graph_nodes = graph_data["nodes"]

    # Build node_name → meta for I/O shape annotation
    node_meta: dict[str, dict[str, Any]] = {
        n["name"]: n.get("meta", {}) for n in graph_nodes
    }

    # User inputs
    inputs: list[dict[str, Any]] = []
    for spec in sig.input_specs:
        if spec.kind != InputKind.USER_INPUT:
            continue
        entry: dict[str, Any] = {"name": spec.arg.name, "kind": "user_input"}
        meta = node_meta.get(spec.arg.name, {})
        if meta.get("shape") is not None:
            entry["shape"] = meta["shape"]
            entry["dtype"] = meta.get("dtype")
        inputs.append(entry)

    # User outputs
    outputs: list[dict[str, Any]] = []
    for spec in sig.output_specs:
        if spec.kind != OutputKind.USER_OUTPUT:
            continue
        entry = {"name": spec.arg.name}
        meta = node_meta.get(spec.arg.name, {})
        if meta.get("shape") is not None:
            entry["shape"] = meta["shape"]
            entry["dtype"] = meta.get("dtype")
        outputs.append(entry)

    # Collect unique ATen ops and find unrecognized ones
    ops_seen: dict[str, int] = {}
    unsupported_ops: list[str] = []
    for node in graph_nodes:
        if node["op"] == "call_function":
            target = node["target"]
            ops_seen[target] = ops_seen.get(target, 0) + 1

    # Manifest graph nodes — enrich with weight_name where applicable
    manifest_nodes: list[dict[str, Any]] = []
    for node in graph_nodes:
        mn: dict[str, Any] = {
            "id": node["id"],
            "name": node["name"],
            "op": node["op"],
            "target": node["target"],
            "args": node["args"],
            "kwargs": node["kwargs"],
            "meta": node["meta"],
        }
        if "kind" in node:
            mn["kind"] = node["kind"]
        if "weight_name" in node:
            mn["weight_name"] = node["weight_name"]
        manifest_nodes.append(mn)

    manifest: dict[str, Any] = {
        "format": "iphso-webgpu-export",
        "format_version": 0,
        "weight_file": "weights.f32.bin",
        "endianness": "little",
        "inputs": inputs,
        "outputs": outputs,
        "weights": weight_entries,
        "graph": {
            "node_count": len(manifest_nodes),
            "op_counts": {k: ops_seen[k] for k in sorted(ops_seen)},
            "nodes": manifest_nodes,
        },
        "warnings": warnings,
        "unsupported_ops": unsupported_ops,
    }

    return manifest


def write_manifest(
    ep: torch.export.ExportedProgram,
    weight_entries: list[dict[str, Any]],
    graph_data: dict[str, Any],
    warnings: list[str],
    out_dir: Path,
) -> dict[str, Any]:
    manifest = build_manifest(ep, weight_entries, graph_data, warnings)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest
