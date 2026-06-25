"""High-level compiler entry points: ExportedProgram -> Package -> .iph file."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.export

from kuma.debug import generate_debug_report
from kuma.export import export_program
from kuma.golden import capture_golden
from kuma.graph import serialize_graph
from kuma.kernels import load_kernels
from kuma.manifest import build_manifest
from kuma.pack_weights import pack_weights
from kuma.package_iph import Package


def compile(ep: torch.export.ExportedProgram, example_inputs: tuple | None = None) -> Package:
    """Compile a captured ExportedProgram into an in-memory Package.

    `example_inputs` is optional and only used to capture golden.json (a real eager run's
    per-node value stats, for verifying a runtime against) -- when omitted (e.g. a caller
    only has the ExportedProgram, not the original inputs), the package simply has no
    golden data, same as before this existed.
    """
    graph_data = serialize_graph(ep)
    weights_blob, weight_entries, skipped = pack_weights(ep)
    warnings = [f"skipped non-float32 tensor: {s}" for s in skipped]
    manifest = build_manifest(ep, weight_entries, graph_data, warnings)
    debug_report = generate_debug_report(ep, graph_data, weight_entries, manifest)
    # Wrapped in "branches" (a single trivial one, no namespacing needed) so this
    # matches kuma.branching.compile_branching's golden.json shape exactly -- one
    # uniform shape for a runtime's verifier to consume either way.
    golden = {"format_version": 0, "branches": [capture_golden(ep, example_inputs)]} if example_inputs is not None else None

    return Package(
        manifest=manifest,
        weights_blob=weights_blob,
        graph_data=graph_data,
        debug_report=debug_report,
        kernels=load_kernels(),
        skipped=skipped,
        golden=golden,
    )


def export_exported_program(
    ep: torch.export.ExportedProgram, out: str | Path, example_inputs: tuple | None = None
) -> Path:
    """Compile an already-captured ExportedProgram and save it as a .iph package."""
    return compile(ep, example_inputs).save(out)


def export_model(
    model: torch.nn.Module,
    example_inputs: tuple,
    out: str | Path,
) -> Path:
    """Run torch.export.export on `model` and save the result as a .iph package."""
    ep = export_program(model, example_inputs)
    return export_exported_program(ep, out, example_inputs)
