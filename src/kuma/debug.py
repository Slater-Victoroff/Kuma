"""Generate the debug report — human-readable summary of the captured graph."""

from __future__ import annotations

from typing import Any

import torch
import torch.export


# Ops we expect to have a WebGPU lowering in a future step.
_LIKELY_SUPPORTED: frozenset[str] = frozenset(
    {
        "aten.convolution.default",
        "aten.add.Tensor",
        "aten.add.Scalar",
        "aten.mul.Tensor",
        "aten.mul.Scalar",
        "aten.relu.default",
        "aten.relu_.default",
        "aten.gelu.default",
        "aten.silu.default",
        "aten.hardswish.default",
        "aten.mm.default",
        "aten.addmm.default",
        "aten.bmm.default",
        "aten.t.default",
        "aten.view.default",
        "aten.reshape.default",
        "aten.permute.default",
        "aten.expand.default",
        "aten.clone.default",
        "aten.cat.default",
        "aten.mean.dim",
        "aten.sum.dim_IntList",
        "aten.native_layer_norm.default",
        "aten.native_batch_norm.default",
        "aten._native_batch_norm_legit_no_training.default",
        "aten.native_group_norm.default",
        "aten.max_pool2d_with_indices.default",
        "aten.avg_pool2d.default",
        "aten._adaptive_avg_pool2d.default",
        "aten.upsample_nearest2d.vec",
        "aten.upsample_bilinear2d.vec",
        "aten.pixel_shuffle.default",
        "aten.clamp.default",
        "aten.hardtanh.default",
        "aten.sigmoid.default",
        "aten.tanh.default",
        "aten.exp.default",
        "aten.log.default",
        "aten.sqrt.default",
        "aten.rsqrt.default",
        "aten.pow.Tensor_Scalar",
        "aten.sub.Tensor",
        "aten.div.Tensor",
        "aten.div.Scalar",
        "aten.neg.default",
        "aten.abs.default",
        "aten.transpose.int",
        "aten.select.int",
        "aten.slice.Tensor",
        "aten.unsqueeze.default",
        "aten.squeeze.dim",
        "aten.flatten.using_ints",
    }
)


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
    unsupported: list[str] = []
    lines += [
        "## ATen Ops Encountered",
        "| Op | Count | Likely WebGPU-ready |",
        "|----|------:|:-------------------:|",
    ]
    for target, count in sorted(ops_seen.items()):
        ready = "yes" if target in _LIKELY_SUPPORTED else "**no**"
        lines.append(f"| `{target}` | {count} | {ready} |")
        if target not in _LIKELY_SUPPORTED:
            unsupported.append(target)
    lines += [""]

    if unsupported:
        lines += ["## Unsupported / Unrecognized Ops"]
        lines.append("These ops have no planned WebGPU lowering yet:")
        for op in sorted(unsupported):
            lines.append(f"- `{op}`")
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
