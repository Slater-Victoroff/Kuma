# Kuma

PyTorch `torch.export` → `.iph` compiler/exporter.

Kuma captures a PyTorch inference graph with `torch.export`, packs its weights into a
contiguous binary blob, and bundles everything into a single self-contained `.iph` package
(a zip file). Kuma is **not** a runtime, **not** a training library, and **not** a WebGPU
framework — `.iph` is just the data contract that a separate WebGPU runtime will eventually
consume.

```
PyTorch Model -> torch.export -> Kuma -> model.iph
```

## Scope (v0)

- Inference only, static shapes, **float32 only**, CPU tensors only.
- Fails loudly (raises `ValueError`/`TypeError`) on float16/bfloat16 inputs or params, non-CPU
  tensors, non-`nn.Module` model factories, or non-tuple example inputs.
- Target ops: Conv2d, Linear, Add, Mul, GELU, ReLU, Reshape, Permute, Concatenate, Slice, plus
  whatever else `torch.export` traces cleanly (BatchNorm/LayerNorm/GroupNorm, pooling, etc. all
  export fine — they just may not have a WGSL kernel yet, see `unsupported_ops` below).
- WGSL kernels are embedded in every package, but there's no WebGPU runtime yet to execute them
  (no fusion, no buffer planning) — that's Step 2+.

## Install

This repo has no local Python/pip — everything is built and tested through Docker
(`docker compose build`, `docker compose run --rm test`). To **use** Kuma from another project,
you don't need any of that; just pip-install the package itself wherever you have Python + torch.

**Editable install from a local checkout (recommended for active dev on both repos):**
```bash
pip install -e /path/to/kuma
```

**Install from git (CI / fixed snapshots):**
```bash
pip install git+https://github.com/yourorg/kuma.git
```

Requires `torch>=2.1` and `numpy>=1.24` (declared as dependencies, will be pulled in
automatically unless you `--no-deps`).

## Python API

```python
import torch
import kuma

model = build_model(cfg)
model.eval()
example_inputs = (torch.randn(1, 3, 512, 512),)   # must match real input shape/dtype exactly

kuma.export_model(model, example_inputs, out="model.iph")
```

### `kuma.export_model(model, example_inputs, out) -> Path`
Runs `torch.export.export` on `model` (after validating dtypes/devices), compiles the result,
and writes a `.iph` package to `out`. This is the one-call happy path.

### `kuma.export_exported_program(ep, out) -> Path`
Same as above, but starting from an `ExportedProgram` you already captured yourself:
```python
ep = torch.export.export(model.eval(), example_inputs)
kuma.export_exported_program(ep, out="model.iph")
```

### `kuma.compile(ep) -> Package`
Lower-level entry point — compiles without writing anything to disk. Useful if you want to
inspect the manifest/weights in-process, or write multiple output forms:
```python
package = kuma.compile(ep)
package.save("model.iph")        # the self-contained .iph zip
package.write_dir("debug/")      # loose files: manifest.json, weights.f32.bin, kernels/,
                                  # debug_report.md, exported_graph.json (debug-dir only)
```

`Package` fields: `manifest` (dict), `weights_blob` (bytes), `graph_data` (dict, the raw FX node
list — not included in the `.iph` zip since `manifest["graph"]` already embeds it),
`debug_report` (str), `kernels` (dict[str, bytes]), `skipped` (list[str] of non-float32 tensors
that were dropped, e.g. BatchNorm's `num_batches_tracked`).

## Wiring up an existing model repo

There's no CLI — write a small script that imports your model and calls the API directly:
```python
# mymodel/export.py
import torch
import kuma
from mymodel.build import build_model

model = build_model(cfg)
model.eval()
example_inputs = (torch.randn(1, 3, 512, 512),)  # match the real input shape/dtype exactly

kuma.export_model(model, example_inputs, out="artifacts/mymodel.iph")
```

## The `.iph` format

A `.iph` file is a zip archive:

| Path | Required | Contents |
|---|---|---|
| `manifest.json` | yes | inputs, outputs, weight table, full graph (nodes/ops/op_counts), warnings |
| `weights.f32.bin` | yes | one contiguous little-endian float32 blob, 4-byte aligned per tensor |
| `kernels/*.wgsl` | no (always present in v0) | embedded WGSL source per op — `add`, `mul`, `gelu`, `relu`, `conv2d`, `linear`, `reshape`, `permute`, `concat`, `slice` |
| `debug_report.md` | no (always present in v0) | human-readable summary: param count, ops table, weight table |

`manifest.json` shape (`format: "kuma"`, `format_version: 0`):
```jsonc
{
  "format": "kuma", "format_version": 0,
  "weight_file": "weights.f32.bin", "endianness": "little",
  "inputs":  [{ "name": "...", "shape": [...], "dtype": "float32" }],
  "outputs": [{ "name": "...", "shape": [...], "dtype": "float32" }],
  "weights": [{ "name": "conv1.weight", "shape": [...], "dtype": "float32",
                "byte_offset": 0, "byte_length": 1728, "n_elements": 432 }],
  "graph": { "node_count": 10, "op_counts": {"aten.convolution.default": 2, ...},
             "nodes": [{ "id", "name", "op", "target", "args", "kwargs", "meta",
                         "kind"?: "parameter"|"buffer"|"user_input", "weight_name"? }] },
  "warnings": ["skipped non-float32 tensor: bn.num_batches_tracked (torch.int64)"],
  "unsupported_ops": []
}
```
Every weight entry's `byte_offset + byte_length` is guaranteed `<= len(weights.f32.bin)`, and
`byte_length == n_elements * 4`. To read a weight back out: slice the blob at
`[byte_offset : byte_offset + byte_length]`, interpret as little-endian float32, reshape to
`shape`.

`graph.nodes` *is* the FX graph — placeholders that map to a parameter/buffer carry a
`weight_name` pointing into the `weights` table; everything else is `call_function`/`output`
nodes with ATen op targets (e.g. `aten.convolution.default`) and JSON-safe args/kwargs (tensor
args become `{"node_ref": "<node name>"}`).

## What this is not yet

No kernel fusion, no buffer/memory planning, no WebGPU runtime to actually execute a `.iph`
package — that's `@kuma/webgpu-runtime`, a separate future project. Kuma's job stops at producing
a correct, self-describing package.
