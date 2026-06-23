import { loadIphPackage } from "./iph/loader.js";
import { requestKumaDevice } from "./gpu/device.js";
import { uploadWeightSlice, uploadFloat32 } from "./gpu/buffers.js";
import { runGraph } from "./engine/scheduler.js";
import type { ResolvedTensor } from "./engine/context.js";
import type { SnippetFn } from "./engine/snippets.js";
import { KumaManifestError } from "./errors.js";
import type { IOSpec, KumaManifest } from "./types/manifest.js";

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
  ) {}

  static async load(source: ArrayBuffer | Response | string): Promise<KumaModel> {
    const { manifest, weights, kernels, snippets } = await loadIphPackage(source);
    const device = await requestKumaDevice();

    const weightBuffers = new Map<string, ResolvedTensor>();
    for (const w of manifest.weights) {
      const buffer = uploadWeightSlice(device, weights, w.byte_offset, w.byte_length);
      weightBuffers.set(w.name, { buffer, shape: w.shape });
    }

    return new KumaModel(device, manifest, kernels, snippets, weightBuffers, new Map(), new Map());
  }

  get inputs(): readonly IOSpec[] {
    return this.manifest.inputs;
  }

  get outputs(): readonly IOSpec[] {
    return this.manifest.outputs;
  }

  async run(inputs: Record<string, Float32Array>): Promise<Record<string, Float32Array>> {
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

    const outputs = await runGraph({
      device: this.device,
      manifest: this.manifest,
      kernels: this.kernels,
      pipelineCache: this.pipelineCache,
      weightBuffers: this.weightBuffers,
      inputBuffers,
      rawInputs,
      snippets: this.snippets,
      snippetCache: this.snippetCache,
    });

    const result: Record<string, Float32Array> = {};
    for (const out of outputs) {
      result[out.name] = out.data;
    }
    return result;
  }
}
