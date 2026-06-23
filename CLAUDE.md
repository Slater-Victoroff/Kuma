# Kuma

PyTorch `torch.export` → `.iph` compiler/exporter (Step 1 of a Torch→WebGPU compiler).

Kuma captures a PyTorch inference graph, packs its weights, and bundles everything into a
self-contained `.iph` package (a zip: `manifest.json`, `weights.f32.bin`, `kernels/*.wgsl`,
`debug_report.md`). Kuma is not a runtime, not a training library, and not a WebGPU framework —
the `.iph` file is the contract between this compiler and the (separate) WebGPU runtime that
will eventually execute it.

## Environment

**There is no local Python or pip on this machine.**
All Python execution, testing, and validation must happen inside Docker.

```bash
# Build image
docker compose build

# Run the acceptance test (simple model export → artifacts/simple.iph + artifacts/simple/)
docker compose run --rm export

# Run the full test suite
docker compose run --rm test

# Run a specific test file
docker compose run --rm test tests/test_convnext_block.py -v

# Or build+run in one shot
docker compose up --build
```

Artifacts land in `./artifacts/` (bind-mounted at `/workspace/artifacts` in the container).

## Testing a code change

1. Run the test suite first:
   ```bash
   docker compose run --rm test
   ```

2. Then do a manual acceptance run to read the artifacts:
   ```bash
   docker compose run --rm export \
     --out /workspace/artifacts/simple.iph \
     --out-dir /workspace/artifacts/simple
   ```

Spot-check the output:
- `artifacts/simple.iph` — the self-contained package (zip); should contain `manifest.json`,
  `weights.f32.bin`, `kernels/*.wgsl`, `debug_report.md`
- `artifacts/simple/debug_report.md` — ops table, weight table, unsupported ops
- `artifacts/simple/manifest.json` — full bundle descriptor; `byte_offset + byte_length` must be consistent
- `artifacts/simple/weights.f32.bin` — size should match sum of `byte_length` fields in manifest
- `artifacts/simple/exported_graph.json` — raw FX node list with `weight_name` on parameter placeholders (debug-dir only; not part of the `.iph` contract since the manifest already embeds the graph)

## Using with a real Niko/Nika model (from another container)

The cleanest integration is to mount this repo into the model container and pip-install it there.

**Option A — bind mount + editable install (recommended for dev):**

In your model repo's `docker-compose.yml`, add:
```yaml
volumes:
  - /home/sl8rv/Projects/Kuma:/kuma
```
Then inside that container (e.g. in an entrypoint or interactively):
```bash
pip install -e /kuma
```

**Option B — install from git (for CI or fixed snapshots):**
```bash
pip install git+https://github.com/yourorg/kuma.git
```

**Direct Python API (the only entry point — there is no CLI):**

```python
import torch
import kuma

model = build_model(cfg)
model.eval()
example_inputs = (torch.randn(1, 3, 512, 512),)

kuma.export_model(model, example_inputs, out="model.iph")
```

or, starting from an already-captured `ExportedProgram`:

```python
ep = torch.export.export(model.eval(), example_inputs)
kuma.export_exported_program(ep, out="model.iph")
```

or, for lower-level access to the in-memory package before writing it out:

```python
package = kuma.compile(ep)
package.save("model.iph")        # self-contained .iph zip
package.write_dir("debug/")      # loose files for inspection
```

**Wiring up an existing model repo:**

Write a thin `export.py` alongside your model code that calls the API directly:
```python
# mymodel/export.py
import torch
import kuma
from mymodel.build import build_model   # your existing factory

model = build_model(cfg)                # fill in your config however you do it
model.eval()
example_inputs = (torch.randn(1, 3, 512, 512),)  # match the exact input shape/dtype your model expects

kuma.export_model(model, example_inputs, out="artifacts/mymodel.iph")
```

## Test suite layout

```
tests/
  conftest.py            # run_pipeline() fixture (kuma.compile -> Package) + shared assertion helpers
  test_package_iph.py    # .iph zip contract + top-level kuma.export_model/export_exported_program/compile API
  test_artifacts.py      # file existence, size consistency, JSON schema, roundtrip
  test_conv.py           # Conv2d variants, depthwise, 1×1, ConvTranspose2d
  test_linear.py         # Linear, MLP, channel-MLP via 1×1 conv
  test_norm.py           # BatchNorm2d, LayerNorm, GroupNorm, ConvNeXt-style norm
  test_activations.py    # GELU, ReLU, SiLU, Hardswish, Sigmoid, Tanh
  test_residual.py       # plain skip, projected skip, bottleneck block
  test_convnext_block.py # full ConvNeXt block (closest to real Niko/Nika target)
  test_error_cases.py    # float16/non-cpu inputs rejected loudly
  test_blocks.py         # SE block, encoder stage, tiny U-Net, multi-head output
  test_concat.py         # torch.cat / skip-connection fusion patterns
  test_depthwise_separable.py  # MobileNet-style depthwise-separable blocks
  test_elementwise.py    # mul, sub, div, clamp, abs
  test_multi_input.py    # multi-tensor model inputs
  test_pooling.py        # MaxPool2d, AvgPool2d, AdaptiveAvgPool2d
  test_shapes.py         # flatten, reshape, permute, transpose, squeeze/unsqueeze
  test_upsample.py       # nn.Upsample, ConvTranspose2d, PixelShuffle
```

## Project layout

```
src/kuma/
  __init__.py       # public API: export_model, export_exported_program, compile, Package
  compiler.py       # compile(ep) -> Package; export_model/export_exported_program
  export.py         # validation + torch.export.export wrapper (export_program)
  graph.py          # FX graph → JSON-friendly dict (serialize_graph)
  pack_weights.py   # Parameters/buffers → contiguous f32 blob (pack_weights)
  manifest.py       # manifest dict builder (build_manifest)
  debug.py          # debug report generator (generate_debug_report)
  package_iph.py    # Package dataclass: .save() writes the .iph zip, .write_dir() writes loose files
  kernels/          # embedded WGSL kernels — elementwise binary/unary ops, matmul/conv
                     # (incl. transposed), reductions, norm (layer/batch/group), pooling
                     # (max/avg/adaptive-avg), upsampling (nearest/bilinear), pixel-shuffle,
                     # and shape ops (reshape/permute/concat/slice) — bundled into every
                     # .iph package. See kernels/__init__.py for the full list.
examples/
  simple.py         # Conv2d → GELU → Conv2d → residual add (acceptance-test model)
  export_simple.py  # Acceptance-test runner (Docker ENTRYPOINT for the `export` service)
artifacts/          # git-ignored; created by the exporter at runtime
```

## Key constraints (Step 1 scope)

- Inference only, static shapes, float32 only.
- `torch.export.export` is the primary capture path.
- WGSL kernels are embedded in every `.iph` package, but there is no WebGPU runtime yet to execute
  them — no kernel fusion or memory/buffer planning yet.
- Fail loudly on unsupported dtypes, non-CPU tensors, or dynamic shapes.
- Target models: Conv/ConvNeXt/operator/decoder style (Niko/Nika family).

## What's next (Step 2+)

- Op decompositions and ATen → WebGPU op lowering beyond the current kernel set
- Buffer planning / memory layout, kernel fusion
- A WebGPU interpreter (`@kuma/webgpu-runtime`) that consumes a `.iph` package
