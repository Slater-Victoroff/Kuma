# Kuma — iphso-webgpu-export

PyTorch → WebGPU graph capture and weight packing library (Step 1 of a Torch→WebGPU compiler).

## Environment

**There is no local Python or pip on this machine.**
All Python execution, testing, and validation must happen inside Docker.

```bash
# Build image
docker compose build

# Run the acceptance test (tiny_model export)
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
     --model examples.tiny_model:create_model \
     --example-input examples.tiny_model:create_example_input \
     --out /workspace/artifacts/tiny
   ```

Spot-check the output:
- `artifacts/tiny/debug_report.md` — ops table, weight table, unsupported ops
- `artifacts/tiny/manifest.json` — full bundle descriptor; `byte_offset + byte_length` must be consistent
- `artifacts/tiny/weights.f32.bin` — size should match sum of `byte_length` fields in manifest
- `artifacts/tiny/exported_graph.json` — raw FX node list with `weight_name` on parameter placeholders

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

**Adapter file in your model repo:**

Write a thin `export_adapter.py` alongside your model code:
```python
# mymodel/export_adapter.py
import torch
from mymodel.build import build_model   # your existing factory

def create_model():
    model = build_model(cfg)            # fill in your config however you do it
    model.eval()
    return model

def create_example_input():
    # match the exact input shape/dtype your model expects
    return (torch.randn(1, 3, 512, 512),)
```

Then run:
```bash
python -m iphso_webgpu_export.cli \
  --model mymodel.export_adapter:create_model \
  --example-input mymodel.export_adapter:create_example_input \
  --out /artifacts/mymodel
```

## Test suite layout

```
tests/
  conftest.py            # run_pipeline() fixture + shared assertion helpers
  test_artifacts.py      # file existence, size consistency, JSON schema, roundtrip
  test_conv.py           # Conv2d variants, depthwise, 1×1, ConvTranspose2d
  test_linear.py         # Linear, MLP, channel-MLP via 1×1 conv
  test_norm.py           # BatchNorm2d, LayerNorm, GroupNorm, ConvNeXt-style norm
  test_activations.py    # GELU, ReLU, SiLU, Hardswish, Sigmoid, Tanh
  test_residual.py       # plain skip, projected skip, bottleneck block
  test_convnext_block.py # full ConvNeXt block (closest to real Niko/Nika target)
  test_error_cases.py    # float16/non-cpu inputs rejected loudly
```

## Project layout

```
src/iphso_webgpu_export/
  cli.py          # Entry point: python -m iphso_webgpu_export.cli
  export.py       # torch.export.export wrapper + validation
  graph.py        # FX graph → exported_graph.json
  pack_weights.py # Parameters/buffers → weights.f32.bin
  manifest.py     # manifest.json builder
  debug.py        # debug_report.md generator
examples/
  tiny_model.py   # Conv2d → GELU → Conv2d → residual add
  export_tiny.py  # Convenience runner
artifacts/        # git-ignored; created by the exporter at runtime
```

## Key constraints (Step 1 scope)

- Inference only, static shapes, float32 only.
- `torch.export.export` is the primary capture path.
- No WGSL generation, no kernel fusion, no memory optimization yet.
- Fail loudly on unsupported dtypes, non-CPU tensors, or dynamic shapes.
- Target models: Conv/ConvNeXt/operator/decoder style (Niko/Nika family).

## What's next (Step 2+)

- Op decompositions and ATen → WebGPU op lowering
- WGSL kernel generation per op
- Buffer planning / memory layout
- WebGPU interpreter that consumes `manifest.json`
