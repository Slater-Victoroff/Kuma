"""Serialize the ExportedProgram's FX graph to exported_graph.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.export
from torch.fx import Node

try:
    from torch.export.graph_signature import InputKind
except ImportError:
    from torch._export.graph_signature import InputKind  # type: ignore[no-redef]


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


def _arg_to_json(arg: Any) -> Any:
    """Recursively convert an FX node argument to a JSON-serializable value."""
    if isinstance(arg, Node):
        return {"node_ref": arg.name}
    if isinstance(arg, (list, tuple)):
        return [_arg_to_json(a) for a in arg]
    if isinstance(arg, dict):
        return {k: _arg_to_json(v) for k, v in arg.items()}
    if isinstance(arg, torch.dtype):
        return _dtype_str(arg)
    if isinstance(arg, torch.device):
        return str(arg)
    if isinstance(arg, torch.memory_format):
        return str(arg)
    if isinstance(arg, (int, float, bool, str)) or arg is None:
        return arg
    return repr(arg)


def _node_meta(node: Node) -> dict[str, Any]:
    """Extract shape/dtype from node.meta['val'] (set by torch.export's fake-tensor pass)."""
    val = node.meta.get("val")
    if val is None:
        return {}
    if isinstance(val, (list, tuple)):
        return {
            "outputs": [
                {"shape": list(v.shape), "dtype": _dtype_str(v.dtype)}
                for v in val
                if hasattr(v, "shape")
            ]
        }
    if hasattr(val, "shape"):
        return {"shape": list(val.shape), "dtype": _dtype_str(val.dtype)}
    return {}


def _target_str(node: Node) -> str:
    t = node.target
    if hasattr(t, "__name__"):
        # Plain callable or ATen OpOverload — both have __name__
        overload = getattr(t, "overloadpacket", None)
        if overload is not None:
            # e.g. torch.ops.aten.convolution.default → "aten.convolution.default"
            return str(t)
        return t.__name__
    return str(t)


def serialize_graph(ep: torch.export.ExportedProgram) -> dict[str, Any]:
    """Build the exported_graph JSON structure from an ExportedProgram."""
    sig = ep.graph_signature

    # node_name → state_dict key for parameter and buffer placeholders
    param_node_map: dict[str, str] = {}
    buffer_node_map: dict[str, str] = {}
    user_input_nodes: set[str] = set()

    for spec in sig.input_specs:
        name = spec.arg.name
        if spec.kind == InputKind.PARAMETER:
            param_node_map[name] = spec.target
        elif spec.kind == InputKind.BUFFER:
            buffer_node_map[name] = spec.target
        elif spec.kind == InputKind.USER_INPUT:
            user_input_nodes.add(name)

    nodes_json: list[dict[str, Any]] = []
    for idx, node in enumerate(ep.graph_module.graph.nodes):
        entry: dict[str, Any] = {
            "id": idx,
            "name": node.name,
            "op": node.op,
            "target": _target_str(node),
            "args": [_arg_to_json(a) for a in node.args],
            "kwargs": _arg_to_json(dict(node.kwargs)),
            "meta": _node_meta(node),
        }

        if node.op == "placeholder":
            if node.name in param_node_map:
                entry["kind"] = "parameter"
                entry["weight_name"] = param_node_map[node.name]
            elif node.name in buffer_node_map:
                entry["kind"] = "buffer"
                entry["weight_name"] = buffer_node_map[node.name]
            else:
                entry["kind"] = "user_input"

        nodes_json.append(entry)

    return {"format_version": 0, "nodes": nodes_json}


def write_graph(ep: torch.export.ExportedProgram, out_dir: Path) -> dict[str, Any]:
    data = serialize_graph(ep)
    (out_dir / "exported_graph.json").write_text(json.dumps(data, indent=2))
    return data
