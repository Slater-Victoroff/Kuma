"""Acceptance-test runner: compile the simple model and write the .iph package + debug dir.

Usage:
    python -m examples.export_simple [--out artifacts/simple.iph] [--out-dir artifacts/simple]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import kuma
from examples.simple import create_example_input, create_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("artifacts/simple.iph"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/simple"))
    args = parser.parse_args()

    model = create_model()
    example_inputs = create_example_input()
    ep = torch.export.export(model, example_inputs)
    package = kuma.compile(ep)

    package.save(args.out)
    package.write_dir(args.out_dir)
    print(f"wrote {args.out} and {args.out_dir}/")


if __name__ == "__main__":
    main()
