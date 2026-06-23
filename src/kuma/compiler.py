"""High-level compiler entry points: ExportedProgram -> Package -> .iph file."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.export

from kuma.debug import generate_debug_report
from kuma.export import export_program
from kuma.graph import serialize_graph
from kuma.kernels import load_kernels
from kuma.manifest import build_manifest
from kuma.pack_weights import pack_weights
from kuma.package_iph import Package


def compile(ep: torch.export.ExportedProgram) -> Package:
    """Compile a captured ExportedProgram into an in-memory Package."""
    graph_data = serialize_graph(ep)
    weights_blob, weight_entries, skipped = pack_weights(ep)
    warnings = [f"skipped non-float32 tensor: {s}" for s in skipped]
    manifest = build_manifest(ep, weight_entries, graph_data, warnings)
    debug_report = generate_debug_report(ep, graph_data, weight_entries, manifest)

    return Package(
        manifest=manifest,
        weights_blob=weights_blob,
        graph_data=graph_data,
        debug_report=debug_report,
        kernels=load_kernels(),
        skipped=skipped,
    )


def export_exported_program(ep: torch.export.ExportedProgram, out: str | Path) -> Path:
    """Compile an already-captured ExportedProgram and save it as a .iph package."""
    return compile(ep).save(out)


def export_model(
    model: torch.nn.Module,
    example_inputs: tuple,
    out: str | Path,
) -> Path:
    """Run torch.export.export on `model` and save the result as a .iph package."""
    ep = export_program(model, example_inputs)
    return export_exported_program(ep, out)
