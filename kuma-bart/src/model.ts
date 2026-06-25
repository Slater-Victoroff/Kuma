import { loadIphPackage } from "./iph/loader.js";
import { requestKumaDevice } from "./gpu/device.js";
import { uploadWeightSlice, uploadFloat32 } from "./gpu/buffers.js";
import { runGraph, type RunGraphOutput } from "./engine/scheduler.js";
import { verifyAgainstGolden, warmUp, type VerifyReport } from "./engine/verify.js";
import { profileGraph, type ProfileReport } from "./engine/profile.js";
import { createBufferPoolState, type BufferPoolState, type ResolvedTensor } from "./engine/context.js";
import type { SnippetFn } from "./engine/snippets.js";
import { KumaManifestError } from "./errors.js";
import type { IOSpec, KumaManifest } from "./types/manifest.js";
import type { GoldenData } from "./types/golden.js";

/** A loaded `.iph` package, ready to run via WebGPU. */
export class KumaModel {
  private constructor(
    private readonly device: GPUDevice,
    private readonly manifest: KumaManifest,
    private readonly kernels: ReadonlyMap<string, string>,
    private readonly snippets: ReadonlyMap<string, string>,
    private readonly weightBuffers: ReadonlyMap<string, ResolvedTensor>,
    private readonly pipelineCache: Map<string, GPUComputePipeline>,
    private readonly snippetCache: Map<string, SnippetFn>,
    private readonly golden: GoldenData | undefined,
    private readonly constantCache: Map<string, GPUBuffer>,
    private readonly bufferPool: BufferPoolState,
  ) {}

  static async load(source: ArrayBuffer | Response | string): Promise<KumaModel> {
    const { manifest, weights, kernels, snippets, golden } = await loadIphPackage(source);
    const device = await requestKumaDevice();

    const weightBuffers = new Map<string, ResolvedTensor>();
    for (const w of manifest.weights) {
      const buffer = uploadWeightSlice(device, weights, w.byte_offset, w.byte_length);
      weightBuffers.set(w.name, { buffer, shape: w.shape });
    }

    return new KumaModel(
      device,
      manifest,
      kernels,
      snippets,
      weightBuffers,
      new Map(),
      new Map(),
      golden,
      new Map(),
      createBufferPoolState(),
    );
  }

  /** Read-only access to the GPU device this model's buffers/pipelines live on -- for
   * a caller (e.g. the demo) that wants to render or otherwise consume a `runRaw()`/
   * `runToGpu()` output's raw `buffer` entirely on-GPU, since WebGPU resources are
   * device-scoped (you can't bind a buffer from this device into a pipeline created on
   * another), or that wants to call `gpuDevice.queue.onSubmittedWorkDone()` itself for
   * backpressure after a `runToGpu()` call. */
  get gpuDevice(): GPUDevice {
    return this.device;
  }

  get inputs(): readonly IOSpec[] {
    return this.manifest.inputs;
  }

  get outputs(): readonly IOSpec[] {
    return this.manifest.outputs;
  }

  private buildInputs(inputs: Record<string, Float32Array>): {
    inputBuffers: Map<string, ResolvedTensor>;
    rawInputs: Map<string, Float32Array>;
  } {
    const inputBuffers = new Map<string, ResolvedTensor>();
    const rawInputs = new Map<string, Float32Array>();
    for (const spec of this.manifest.inputs) {
      const data = inputs[spec.name];
      if (!data) {
        throw new KumaManifestError(
          `Missing input "${spec.name}" — expected one of: ${this.manifest.inputs.map((s) => s.name).join(", ")}`,
        );
      }
      const expected = spec.shape ? spec.shape.reduce((a, b) => a * b, 1) : data.length;
      if (data.length !== expected) {
        throw new KumaManifestError(
          `Input "${spec.name}" expected ${expected} elements (shape ${JSON.stringify(spec.shape)}), got ${data.length}.`,
        );
      }
      rawInputs.set(spec.name, data);
      const buffer = uploadFloat32(this.device, data);
      inputBuffers.set(spec.name, { buffer, shape: spec.shape ?? [data.length] });
    }
    return { inputBuffers, rawInputs };
  }

  private runGraphWith(inputs: Record<string, Float32Array>, skipOutputReadback: boolean): Promise<RunGraphOutput[]> {
    const { inputBuffers, rawInputs } = this.buildInputs(inputs);
    return runGraph({
      device: this.device,
      manifest: this.manifest,
      kernels: this.kernels,
      pipelineCache: this.pipelineCache,
      weightBuffers: this.weightBuffers,
      inputBuffers,
      rawInputs,
      snippets: this.snippets,
      snippetCache: this.snippetCache,
      constantCache: this.constantCache,
      bufferPool: this.bufferPool,
      skipOutputReadback,
    });
  }

  /** Like `run`, but returns the full RunGraphOutput[] (shape + the GPU buffer the
   * data was read back from, not just the data) -- for a caller that wants to render
   * or otherwise consume a result on-GPU instead of paying for a readback it doesn't
   * actually need data for. */
  async runRaw(inputs: Record<string, Float32Array>): Promise<RunGraphOutput[]> {
    return this.runGraphWith(inputs, false);
  }

  /** Like `runRaw`, but skips the GPU->CPU readback for the model's own outputs
   * entirely -- `data` on the result is an empty Float32Array(0); use `buffer` (e.g.
   * to render directly, which the GPU correctly orders after this call's own
   * submission with no CPU-side wait needed). Measured the readback itself
   * (mapAsync'ing a multi-megabyte frame back into JS) at 60-70ms out of a ~65-75ms
   * total per-frame time for a 720x1280x3 model -- i.e. it dominated total latency far
   * more than the CPU-side cost of encoding every dispatch (~2ms). Skip it entirely
   * for an interactive path (scrubbing a slider, playback) that only needs pixels on
   * screen, not values back in JS. For real GPU compute time, see profile()'s
   * realMilliseconds, not totalMicroseconds (which measures something else -- see its
   * own docs). */
  async runToGpu(inputs: Record<string, Float32Array>): Promise<RunGraphOutput[]> {
    return this.runGraphWith(inputs, true);
  }

  async run(inputs: Record<string, Float32Array>): Promise<Record<string, Float32Array>> {
    const outputs = await this.runRaw(inputs);
    const result: Record<string, Float32Array> = {};
    for (const out of outputs) {
      result[out.name] = out.data;
    }
    return result;
  }

  /** Runs the same graph a real `run(inputs)` call would, but with GPU timestamp
   * queries around every dispatched node instead of one shared compute pass -- see
   * engine/profile.ts for what that does and doesn't distort (short version: its
   * totalMicroseconds runs 3-4x *higher* than real frame time, a hard consequence of
   * WebGPU's timestampWrites being pass-scoped, not a bug -- treat byTarget's
   * percentages as the signal). `realMilliseconds` below is measured the same way the
   * interactive path is (skip the output readback, wait for actual GPU completion via
   * onSubmittedWorkDone instead of mapAsync) so there's one trustworthy number to set
   * expectations against, in the same report. `baselineMilliseconds` measures
   * onSubmittedWorkDone() with *nothing newly submitted* -- isolates whatever fixed
   * latency that call itself has in this browser's implementation, independent of how
   * much GPU work was actually done, since totalMicroseconds and realMilliseconds can
   * diverge by far more than per-pass overhead alone explains once real GPU work gets
   * small enough (a few ms) for a fixed sync-call cost to dominate instead. Throws if
   * this device/browser doesn't support the 'timestamp-query' WebGPU feature. */
  async profile(
    inputs: Record<string, Float32Array>,
  ): Promise<ProfileReport & { realMilliseconds: number; baselineMilliseconds: number }> {
    const { inputBuffers, rawInputs } = this.buildInputs(inputs);
    const report = await profileGraph({
      device: this.device,
      manifest: this.manifest,
      kernels: this.kernels,
      pipelineCache: this.pipelineCache,
      weightBuffers: this.weightBuffers,
      inputBuffers,
      rawInputs,
      snippets: this.snippets,
      snippetCache: this.snippetCache,
      constantCache: this.constantCache,
      bufferPool: this.bufferPool,
    });

    const baselineT0 = performance.now();
    await this.device.queue.onSubmittedWorkDone();
    const baselineMilliseconds = performance.now() - baselineT0;

    const t0 = performance.now();
    await this.runGraphWith(inputs, true);
    await this.device.queue.onSubmittedWorkDone();
    const realMilliseconds = performance.now() - t0;

    return { ...report, realMilliseconds, baselineMilliseconds };
  }

  /** Runs every branch directly against golden.json's own recorded reference input
   * (bypassing the time-based router entirely) and compares the actual computed values
   * against what a real eager-PyTorch run produced -- see engine/verify.ts. Throws if
   * this package wasn't exported with golden.json (i.e. the exporter had no example
   * inputs to capture it with). */
  async verify(): Promise<VerifyReport> {
    if (!this.golden) {
      throw new KumaManifestError(
        "This .iph package has no golden.json -- it was exported without example inputs to capture golden values with.",
      );
    }
    return verifyAgainstGolden(
      {
        device: this.device,
        kernels: this.kernels,
        pipelineCache: this.pipelineCache,
        weightBuffers: this.weightBuffers,
        snippets: this.snippets,
        snippetCache: this.snippetCache,
        constantCache: this.constantCache,
        bufferPool: this.bufferPool,
      },
      this.manifest,
      this.golden,
    );
  }

  /** Runs every branch once, directly against golden.json's own recorded reference
   * input (bypassing the router, same mechanism as verify()) -- purely to populate
   * pipelineCache/constantCache for every kernel/shape every branch could possibly
   * need, before the caller starts interacting with the model. Without this, the
   * *first* time a given branch (or a kernel-routing decision within it) actually
   * gets exercised during interactive use -- e.g. scrubbing across a segment boundary
   * into a branch never yet rendered -- pays for real shader compilation right then, a
   * genuine stutter landing at exactly the worst moment. Intended to be called once,
   * right after load(), before enabling any interactive controls. A no-op (not a
   * throw, unlike verify()) when this package has no golden.json -- nothing to warm up
   * ahead of time that a first real run() call wouldn't do anyway, since there's only
   * the one graph. */
  async warmUp(): Promise<void> {
    if (!this.golden) return;
    await warmUp(
      {
        device: this.device,
        kernels: this.kernels,
        pipelineCache: this.pipelineCache,
        weightBuffers: this.weightBuffers,
        snippets: this.snippets,
        snippetCache: this.snippetCache,
        constantCache: this.constantCache,
        bufferPool: this.bufferPool,
      },
      this.manifest,
      this.golden,
    );
  }
}
