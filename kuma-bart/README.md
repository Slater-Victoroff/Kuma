# kuma-bart

"BAd RunTime for Kuma" — a WebGPU interpreter that loads a `.iph` package (produced by
the Python `kuma` compiler, see the [root README](../README.md)) and actually runs it,
live, in the browser.

```
model.iph -> kuma-bart (WebGPU) -> pixels in a <canvas>
```

## What it does

- Loads a `.iph` package (manifest + packed weights + embedded WGSL kernels, optionally
  `golden.json`) and uploads weights to GPU buffers once at load time.
- Walks the manifest's graph in topological order, dispatching one WebGPU compute kernel
  per node, sharing a single command encoder/compute pass per inference call.
- Covers the op surface Kuma's compiler currently exports: conv2d (general kernel plus
  depthwise/pointwise-specialized fast paths), linear, norm (batch/layer/group),
  activations, pooling, upsampling, reshape/permute/concat/slice, and the
  complex/FFT/einsum/Tucker-style ops used by decoder-style models — plus branching
  (`switch`/`js_snippet`-routed multi-segment models, e.g. a time-sliced animation).
- Verifies its own output against `golden.json` (recorded eager-PyTorch reference
  values the Python exporter embeds in the package) — `model.verify()`.
- Renders directly from a GPU storage buffer to a `<canvas>`, with no CPU round-trip.

## Status

Functional, not yet fast in the way the eventual goal is, but real: a 4-branch,
~270-node ConvNeXt-style model plays back live in a browser at full framerate. There is
no kernel fusion pass and no general buffer/memory planner yet — Kuma's `.iph` format
doesn't carry the information a real planner would need (that's Step 2+ work; see root
`CLAUDE.md`). What's here instead: a handful of hand-specialized kernels (tiled matmul,
halo-tiled depthwise conv, channel-mixing pointwise conv, parallel-reduction norm) and
runtime infrastructure (buffer pooling, multi-frame in-flight pipelining, pipeline
pre-warming) that make a real model usable interactively, not just numerically correct.

## Build / test / demo

Two ways to run this, mirroring the Python side:

**Via Docker, from the repo root (recommended — matches `docker-compose.yml`):**
```bash
docker compose run --rm bart-test    # typecheck + vitest
docker compose up bart-demo          # serves demo/ at localhost:5173
```
`bart-demo` bind-mounts the repo root's `artifacts/` directory read-only at
`demo/artifacts/` — the same place the Python `export` service writes to. So the usual
loop is: `docker compose run --rm export` (writes `artifacts/simple.iph`), then open the
demo and load `/artifacts/simple.iph`.

**Directly with npm (needs a real Node 20+ — the system Node on some dev boxes is too
old for vite/vitest to run at all):**
```bash
cd kuma-bart
npm install
npm run test     # typecheck + vitest
npm run demo     # vite dev server (demo/ as root)
npm run build    # library build -> dist/
```

## Using it from your own code

```ts
import { KumaModel } from "kuma-bart";

const model = await KumaModel.load("/artifacts/simple.iph"); // path, Response, or ArrayBuffer
await model.warmUp(); // optional, recommended before any interactive use -- see below

const outputs = await model.run({ input: someFloat32Array }); // CPU-readback result

// Interactive/rendering path: skip the CPU readback, consume the result on-GPU.
const raw = await model.runToGpu({ input: someFloat32Array });
// raw[0].buffer is a GPUBuffer on model.gpuDevice, ready to bind into your own render
// pass (see demo/gpuRender.ts for a minimal fullscreen-quad example).

const report = await model.verify();                       // vs. golden.json, if present
const profile = await model.profile({ input: someFloat32Array }); // per-op GPU timing

// Optional, model-author-provided playback metadata (see the root README's .iph
// format section) -- undefined for any .iph exported without it. A caller scrubbing/
// playing a normalized [0,1] time input should use duration_seconds instead of
// assuming one fixed speed for every model.
model.playback?.duration_seconds; // -> number | undefined
```

## Architecture

```
src/
  index.ts            public API surface (KumaModel, error types, manifest/golden/op types)
  model.ts            KumaModel: load(), run/runRaw/runToGpu, profile, verify, warmUp
  errors.ts           KumaManifestError / KumaShapeError / KumaUnsupportedOpError
  iph/loader.ts        unzips a .iph: manifest.json, weights blob, kernels/*.wgsl, golden.json

  engine/
    scheduler.ts        runGraph -- the core interpreter loop: topological walk, dispatch,
                         readback, end-of-call buffer cleanup
    context.ts           OpContext -- per-node GPU mechanics handed to every op handler
                         (buffer pooling, params caching, pipeline cache, dispatch helpers)
    profile.ts           profileGraph -- one GPU timestamp-scoped compute pass per node,
                         for a per-op timing breakdown (see its own docs for what that
                         does and doesn't distort vs. a real shared-pass run)
    verify.ts            golden-value verification + warmUp, both by running each switch
                         branch directly against its own recorded reference input,
                         bypassing the time-based router entirely
    dispatch.ts          workgroup grid math shared by dispatchKernel/dispatchKernelGrid
    snippets.ts          sandboxed eval of js_snippet graph nodes (branch-routing logic)
    dft.ts               DFT basis matrix construction for the FFT-family ops
    einsum-plan.ts        einsum subscript string -> concrete dispatch plan
    shape.ts / stats.ts   shape/stride helpers; tensor summary stats used by verify

  ops/                  one file per op family; each exports an OpHandler
                         (ctx: OpContext) => void registered in ops/index.ts's opRegistry

  gpu/
    device.ts            WebGPU adapter/device request + feature detection (e.g.
                         "timestamp-query", needed by profile())
    buffers.ts            buffer creation/upload/readback helpers
    params.ts             uniform buffer packing (u32-only and mixed u32/f32 Params structs)

  types/                 manifest.json / golden.json TS types, matching the Python
                         side's schema exactly

demo/
  index.html / main.ts   interactive browser demo: load a .iph, scrub a time slider,
                         Play / Verify / Profile
  gpuRender.ts            GPU-direct canvas rendering (no CPU pixel round-trip)

test/                    vitest unit tests for the pure-logic pieces (dispatch grid math,
                         params packing, shape utils, snippet eval, einsum planning, iph
                         loading) plus mock-gpu.ts, a mock GPUDevice for testing scheduler
                         logic without a real browser/adapter
```

## Key design decisions worth knowing before changing this

### Buffer pooling and the multi-frame-in-flight invariant

`OpContext.createBuffer` doesn't allocate a fresh GPU buffer on every call — every
node's output comes from a small (`BUFFER_POOL_DEPTH`-deep) ring buffer, reused forever
once allocated. This relies on Kuma being a **static-shape** compiler: the same node
produces the same shape on every single call, so the same buffer is always the right
size. Without this, a real model (~270 dispatched nodes) was allocating and immediately
destroying hundreds of GPUBuffer/bind-group/uniform-buffer JS objects every frame at up
to 60fps — enough sustained garbage to trigger real V8 GC pauses, confirmed directly via
Chrome DevTools Performance recordings (GC surges landing exactly where dropped frames
showed up on the Frames track). See `engine/context.ts`'s `BufferPoolState`/
`acquireGenerationSlot`/`releaseGenerationSlot` for the generation-counter scheme that
keeps reuse safe under multi-frame pipelining: WebGPU exposes no per-resource fence,
only a queue-wide `onSubmittedWorkDone()`, so a buffer slot is never handed out again
until that's confirmed for its previous occupant — checked once per call, and only
actually awaited when the pool is genuinely exhausted, not on every call.

### The ~30ms GPU-completion-notification latency

`device.queue.onSubmittedWorkDone()` / `buffer.mapAsync()` have a real, measured ~30ms
latency in at least one common browser's WebGPU implementation to report that submitted
work is done — independent of how much GPU work or data is actually involved (measured:
a 16-byte debug readback took the same ~30ms as a real 11MB frame; a GPU-side timestamp
query wrapped directly around the compute pass showed true GPU execution at ~7ms,
matching `profile()`'s per-node-summed breakdown almost exactly). This is a browser/
driver characteristic, not a kernel or scheduling bug in this codebase — the fix is
pipelining multiple frames in flight so the latency overlaps with useful work instead of
being paid serially per frame (see the in-flight submission pattern in `demo/main.ts`).

### Pipeline/shader warm-up

`KumaModel.warmUp()` runs every switch branch once (against `golden.json`'s own
reference input, bypassing the router) purely to populate the pipeline cache and buffer
pool ahead of interactive use. Without it, the first time a branch — or a
kernel-routing decision within one, e.g. conv2d's depthwise/pointwise/general split —
gets exercised during real playback, it pays for shader compilation right then: a real,
user-visible stutter landing at the worst possible moment.

### `skipOutputReadback` / `runToGpu`

Reading a multi-megabyte output buffer back to the CPU (`mapAsync`) measured at
60-70ms — far more than the GPU compute itself for a typical frame. A caller that only
needs to render the result (which can stay entirely on-GPU — see `gpuRender.ts`'s
direct-from-storage-buffer render pass) should use `runToGpu()`/`runRaw()`, not `run()`.

## Known gaps

- No general buffer/memory planner: the pool above is a fixed-depth reuse ring per
  node, not an allocator that reasons about a graph's actual live ranges.
- No automatic kernel fusion pass. A few op-specific specializations exist (depthwise/
  pointwise conv2d, tiled linear, parallel-reduction groupnorm) but nothing general.
- `ConvTranspose2d` (transposed convolution) has no WGSL kernel yet — throws
  `KumaUnsupportedOpError` if a manifest needs one.
