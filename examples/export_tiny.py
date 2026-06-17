"""Convenience runner: export the tiny model from the repo root.

Usage:
    python examples/export_tiny.py [--out artifacts/tiny]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/tiny", help="Output directory")
    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "iphso_webgpu_export.cli",
        "--model", "examples.tiny_model:create_model",
        "--example-input", "examples.tiny_model:create_example_input",
        "--out", args.out,
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1])
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
