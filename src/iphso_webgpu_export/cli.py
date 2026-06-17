"""CLI entry point: python -m iphso_webgpu_export.cli [options]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="iphso-webgpu-export",
        description="Export a PyTorch model to a WebGPU artifact directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m iphso_webgpu_export.cli \\
      --model examples.tiny_model:create_model \\
      --example-input examples.tiny_model:create_example_input \\
      --out artifacts/tiny
""",
    )
    parser.add_argument(
        "--model",
        required=True,
        metavar="MODULE:FACTORY",
        help="'dotted.module:function' returning an nn.Module (no-arg callable).",
    )
    parser.add_argument(
        "--example-input",
        required=True,
        dest="example_input",
        metavar="MODULE:FACTORY",
        help="'dotted.module:function' returning a tuple[Tensor, ...] (no-arg callable).",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        metavar="DIR",
        help="Output directory; created if it does not exist.",
    )

    args = parser.parse_args()
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Make the project root (CWD) importable so 'examples.tiny_model' resolves.
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    # Late imports so torch doesn't load until we've parsed args.
    from iphso_webgpu_export.export import run_export
    from iphso_webgpu_export.graph import write_graph
    from iphso_webgpu_export.pack_weights import write_weights
    from iphso_webgpu_export.manifest import write_manifest
    from iphso_webgpu_export.debug import write_debug_report

    print(f"[export]   model        : {args.model}")
    print(f"[export]   example input: {args.example_input}")
    print(f"[export]   output dir   : {out_dir.resolve()}")
    print()

    print("[export]   running torch.export.export ...")
    ep = run_export(
        model_spec=args.model,
        input_spec=args.example_input,
        out_dir=out_dir,
    )
    print("[export]   done.\n")

    print("[graph]    serializing FX graph ...")
    graph_data = write_graph(ep, out_dir)
    print(f"[graph]    {len(graph_data['nodes'])} nodes → exported_graph.json\n")

    print("[weights]  packing float32 tensors ...")
    blob, weight_entries, skipped = write_weights(ep, out_dir)
    total_bytes = sum(w["byte_length"] for w in weight_entries)
    print(
        f"[weights]  {len(weight_entries)} tensors, {total_bytes:,} bytes → weights.f32.bin"
    )
    if skipped:
        print(f"[weights]  skipped (non-float32): {', '.join(skipped)}")
    print()

    print("[manifest] building manifest ...")
    warnings: list[str] = [f"skipped non-float32 tensor: {s}" for s in skipped]
    manifest = write_manifest(ep, weight_entries, graph_data, warnings, out_dir)
    print(f"[manifest] {len(manifest['graph']['op_counts'])} unique ops → manifest.json\n")

    print("[debug]    writing debug report ...")
    write_debug_report(ep, graph_data, weight_entries, manifest, out_dir)
    print("[debug]    debug_report.md\n")

    print("=" * 56)
    print("  Export complete")
    print("=" * 56)
    for fname in ["exported_graph.json", "weights.f32.bin", "manifest.json", "debug_report.md"]:
        p = out_dir / fname
        size = p.stat().st_size if p.exists() else 0
        print(f"  {fname:<30}  {size:>10,} bytes")
    print()


if __name__ == "__main__":
    main()
