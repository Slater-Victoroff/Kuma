# kuma-bart

WebGPU runtime for `.iph` packages produced by the `kuma` Python compiler. For what this project is and how the pipeline works, see the [root README](../README.md). This file covers the runtime internals and design decisions — useful if you're changing something and want to know why it's the way it is.

## Build / test / demo

```bash
# From repo root (recommended)
docker compose run --rm bart-test    # typecheck + vitest
docker compose up bart-demo          # serves demo/ at localhost:5173

# Or directly with npm (requires Node 20+)
cd kuma-bart && npm install
npm run test     # typecheck + vitest
npm run demo     # vite dev server
npm run build    # library build → dist/
```

`bart-demo` bind-mounts `artifacts/` read-only at `demo/artifacts/`. Run `docker compose run --rm export` first to populate it, then select a model from the dropdown.

## Architecture

```
src/
  index.ts            Public API (KumaModel, error types, manifest/golden/op types)
  model.ts            KumaModel: load, run/runRaw/runToGpu, verify, profile, warmUp
  errors.ts           KumaManifestError / KumaShapeError / KumaUnsupportedOpError
  iph/loader.ts       Unzips a .iph: manifest.json, weights blob, kernels/*.wgsl, golden.json

  player/
    KumaPlayer.ts     <kuma-player> web component (see root README for usage)
    gpuRender.ts      GPU-direct canvas rendering (no CPU pixel round-trip)

  engine/
    scheduler.ts      runGraph — topological walk, per-node dispatch, readback, cleanup
    context.ts        OpContext — per-node GPU mechanics (buffer pool, pipeline cache,
                      dispatch helpers) handed to every op handler
    profile.ts        profileGraph — GPU timestamp-scoped per-op timing breakdown
    verify.ts         Golden-value verification + warmUp
    dispatch.ts       Workgroup grid math
    snippets.ts       Sandboxed eval of js_snippet nodes (branch-routing logic)
    dft.ts            DFT basis matrix for FFT-family ops
    einsum-plan.ts    Einsum subscript → concrete dispatch plan
    shape.ts          Shape/stride helpers
    stats.ts          Tensor summary stats used by verify

  ops/                One file per op family, each exporting an OpHandler registered
                      in ops/index.ts's opRegistry

  gpu/
    device.ts         WebGPU adapter/device init + feature detection
    buffers.ts        Buffer creation, upload, readback helpers
    params.ts         Uniform buffer packing (u32-only and mixed u32/f32 structs)

  types/              TypeScript types for manifest.json and golden.json

test/                 vitest unit tests for pure-logic pieces + mock-gpu.ts
demo/                 Interactive player (thin shell around <kuma-player>)
```

## Key design decisions

### Buffer pooling

`OpContext.createBuffer` doesn't allocate a fresh GPU buffer on every call. Each node gets a `BUFFER_POOL_DEPTH`-deep ring of pre-allocated buffers, reused forever. This works because kuma is a static-shape compiler — the same node always produces the same shape, so the same buffer is always the right size.

Without pooling, a ~270-node model at 60fps allocates and immediately destroys hundreds of `GPUBuffer`/bind-group/uniform objects per frame. That's enough sustained pressure to trigger real V8 GC pauses — confirmed via Chrome DevTools Performance recordings where GC surges correlated directly with frame drops.

A generation counter (`acquireGenerationSlot`/`releaseGenerationSlot` in `context.ts`) keeps reuse safe under multi-frame pipelining: a buffer slot isn't handed out again until `onSubmittedWorkDone()` confirms the previous frame using it has completed. That wait only actually happens when the pool is genuinely exhausted — it's not paid on every call.

### Multi-frame pipelining

`device.queue.onSubmittedWorkDone()` has a measured ~30ms latency in at least one common browser WebGPU implementation — independent of how much GPU work is involved. A GPU-side timestamp around a real inference pass showed ~7ms of actual compute; the same 30ms appeared on a 16-byte debug readback.

Waiting for that signal after every frame serializes 30ms of fixed overhead per frame, which caps throughput at ~33fps regardless of GPU speed. Allowing `BUFFER_POOL_DEPTH` frames in flight simultaneously lets that latency overlap with submission of the next frame. The demo-level `MAX_IN_FLIGHT` cap matches `BUFFER_POOL_DEPTH` exactly — submitting more frames than the pool depth doesn't buy more real concurrency, since the library's own `acquireGenerationSlot` would block internally anyway.

### `runToGpu` / skipping the CPU readback

Reading a multi-megabyte output buffer back to the CPU (`mapAsync`) measured at 60–70ms on real hardware — far more than the GPU compute itself. A caller that only needs to render the result (which can stay entirely on GPU — see `player/gpuRender.ts` for a direct-from-storage-buffer render pass) should call `runToGpu()`, not `run()`.

### Pipeline warm-up

The first use of a compute pipeline triggers WebGPU shader compilation. For a multi-branch model, that happens the first time each branch gets exercised during real playback — landing as a stutter exactly at segment boundaries.

`model.warmUp()` runs every switch branch once (against `golden.json`'s reference inputs, bypassing the time-based router) to populate the pipeline cache and buffer pool ahead of interactive use. Shader compilation happens during the already-expected load wait.

## Known gaps

- No general memory planner. The buffer pool is a fixed-depth reuse ring per node, not an allocator that reasons about a graph's actual live ranges.
- No automatic kernel fusion. A few op-specific specializations exist (tiled matmul, halo-tiled depthwise conv, parallel-reduction groupnorm) but nothing that crosses op boundaries.
- `ConvTranspose2d` kernel is missing — throws `KumaUnsupportedOpError` if a manifest needs one.

Both the planner and fusion need new information in the `.iph` format (live-range annotations, fusable-subgraph markers) that the compiler doesn't emit yet. That's Step 2 work on both sides of the pipeline.
