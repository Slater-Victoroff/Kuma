"""Generate the debug report — human-readable summary of the captured graph."""

from __future__ import annotations

from typing import Any

import torch
import torch.export


def _total_params(ep: torch.export.ExportedProgram) -> int:
    total = sum(t.numel() for t in ep.state_dict.values())
    constants = getattr(ep, "constants", None) or {}
    for val in constants.values():
        if isinstance(val, torch.Tensor):
            total += val.numel()
    return total


def generate_debug_report(
    ep: torch.export.ExportedProgram,
    graph_data: dict[str, Any],
    weight_entries: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> str:
    lines: list[str] = []

    lines += ["# Kuma — Debug Report", ""]

    # ── Summary ──────────────────────────────────────────────────────────────
    total_params = _total_params(ep)
    total_bytes = sum(w["byte_length"] for w in weight_entries)
    n_nodes = len(graph_data["nodes"])
    ops_seen: dict[str, int] = manifest["graph"]["op_counts"]

    lines += [
        "## Summary",
        f"- Total parameters : {total_params:,}",
        f"- Weight blob size : {total_bytes:,} bytes  ({total_bytes / 1024 / 1024:.4f} MB)",
        f"- Graph nodes      : {n_nodes}",
        f"- Unique ops       : {len(ops_seen)}",
        "",
    ]

    # ── Inputs / Outputs ─────────────────────────────────────────────────────
    lines += ["## Inputs"]
    for inp in manifest["inputs"]:
        lines.append(f"- `{inp['name']}` : shape={inp.get('shape')}  dtype={inp.get('dtype')}")
    lines += ["", "## Outputs"]
    for out in manifest["outputs"]:
        lines.append(f"- `{out['name']}` : shape={out.get('shape')}  dtype={out.get('dtype')}")
    lines += [""]

    # ── ATen ops ─────────────────────────────────────────────────────────────
    # Whether kuma-bart (the WebGPU runtime, see ../../kuma-bart/README.md) actually
    # has a kernel for a given op isn't something this report can determine -- Python
    # has no way to introspect a TypeScript module, and a hand-maintained guess here
    # would only ever drift out of sync with kuma-bart's real op coverage (which it
    # did, for a long time, before being removed). kuma-bart's own opRegistry fails
    # loudly (KumaUnsupportedOpError) at runtime for anything it doesn't actually
    # support -- that's the authoritative answer, not a static list in the compiler.
    lines += [
        "## ATen Ops Encountered",
        "| Op | Count |",
        "|----|------:|",
    ]
    for target, count in sorted(ops_seen.items()):
        lines.append(f"| `{target}` | {count} |")
    lines += [""]

    # ── Weights ──────────────────────────────────────────────────────────────
    lines += [
        "## Weights",
        "| Name | Shape | Elements | Bytes | Offset |",
        "|------|-------|--------:|------:|-------:|",
    ]
    for w in weight_entries:
        lines.append(
            f"| `{w['name']}` | {w['shape']} |"
            f" {w['n_elements']:,} | {w['byte_length']:,} | {w['byte_offset']:,} |"
        )
    lines += [""]

    # ── Graph nodes ──────────────────────────────────────────────────────────
    lines += [
        "## Graph Nodes",
        "| ID | Name | Op | Target | Shape | Dtype |",
        "|----|------|----|--------|-------|-------|",
    ]
    for node in graph_data["nodes"]:
        meta = node.get("meta", {})
        shape = meta.get("shape", "")
        dtype = meta.get("dtype", "")
        lines.append(
            f"| {node['id']} | `{node['name']}` | {node['op']} |"
            f" `{node['target']}` | {shape} | {dtype} |"
        )
    lines += [""]

    # ── Warnings ─────────────────────────────────────────────────────────────
    all_warnings = manifest.get("warnings", [])
    if all_warnings:
        lines += ["## Warnings"]
        for w in all_warnings:
            lines.append(f"- {w}")
        lines += [""]

    return "\n".join(lines)
