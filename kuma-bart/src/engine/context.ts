import { isNodeRef, type ArgValue, type GraphNode } from "../types/manifest.js";
import { createStorageBuffer, createUniformBuffer, uploadFloat32 } from "../gpu/buffers.js";
import { packParams, packTypedParams, type TypedParamField } from "../gpu/params.js";
import { KumaShapeError, KumaManifestError } from "../errors.js";
import { numElements } from "./shape.js";
import { computeDispatchGrid, KERNEL_WORKGROUP_SIZE, MAX_WORKGROUPS_PER_DIMENSION } from "./dispatch.js";

export interface ResolvedTensor {
  buffer: GPUBuffer;
  shape: number[];
  /** Set when this tensor logically represents a complex64 value — `buffer` holds the
   * real part, `imag` the imaginary part, paired up purely on the JS side by
   * aten.complex.default (see ops/complex.ts). No kernel ever sees an interleaved
   * complex buffer; real/imag are always two separate same-shape real GPUBuffers. */
  imag?: GPUBuffer;
}

/**
 * Per-node execution context handed to an op handler. Wraps the GPU mechanics
 * (pipeline caching, bind groups, dispatch) so handlers only deal with tensors and
 * shapes. Every kernel binds its buffers in the exact order declared in its WGSL
 * source (storage buffers first, uniform Params last) — `dispatchKernel` relies on
 * callers passing buffers in that same order.
 */
export class OpContext {
  constructor(
    public readonly device: GPUDevice,
    private readonly kernels: ReadonlyMap<string, string>,
    private readonly pipelineCache: Map<string, GPUComputePipeline>,
    private readonly pass: GPUComputePassEncoder,
    private readonly resolved: Map<string, ResolvedTensor>,
    public readonly node: GraphNode,
  ) {}

  resolve(arg: ArgValue): ResolvedTensor {
    if (!isNodeRef(arg)) {
      throw new KumaShapeError(
        `Op "${this.node.target}" (node "${this.node.name}") expected a tensor reference, got ${JSON.stringify(arg)}.`,
      );
    }
    const tensor = this.resolved.get(arg.node_ref);
    if (!tensor) {
      throw new KumaShapeError(
        `Node "${this.node.name}" references "${arg.node_ref}" before it was computed — manifest graph is not topologically ordered.`,
      );
    }
    return tensor;
  }

  /** Allocate a fresh, zero-initialized storage buffer sized for `shape` (no pooling/reuse in v1). */
  createBuffer(shape: readonly number[]): GPUBuffer {
    return createStorageBuffer(this.device, numElements(shape) * 4);
  }

  /** A zero-filled buffer of `byteLength` bytes — for bias-less ops (e.g. conv2d/linear with no bias arg). */
  zeros(byteLength: number): GPUBuffer {
    return createStorageBuffer(this.device, byteLength);
  }

  /** Upload a JS-computed constant (e.g. a DFT basis matrix — see engine/dft.ts) as a
   * storage buffer, ready to bind as a kernel input like any other tensor. */
  uploadConstant(data: Float32Array): GPUBuffer {
    return uploadFloat32(this.device, data);
  }

  /** Pack u32 fields (WGSL struct order) into a ready-to-bind uniform buffer. */
  uniform(fields: readonly number[]): GPUBuffer {
    return createUniformBuffer(this.device, packParams(fields));
  }

  /** Like `uniform`, for Params structs that mix u32 and f32 fields (e.g. clamp,
   * group_norm/layer_norm's eps). */
  uniformTyped(fields: readonly TypedParamField[]): GPUBuffer {
    return createUniformBuffer(this.device, packTypedParams(fields));
  }

  /** Register this node's computed result so later nodes can resolve it by name. */
  setOutput(buffer: GPUBuffer, shape: readonly number[], imag?: GPUBuffer): void {
    this.resolved.set(this.node.name, { buffer, shape: [...shape], imag });
  }

  /** Register an already-built ResolvedTensor as this node's result (e.g. a free
   * passthrough that reuses an existing buffer, or a complex pairing). */
  setOutputTensor(tensor: ResolvedTensor): void {
    this.resolved.set(this.node.name, tensor);
  }

  /** For multi-output nodes (e.g. aten.chunk.default): register one of several results
   * under a synthetic per-index key, later picked up by a getitem(thisNode, i) node. */
  setIndexedOutput(index: number, tensor: ResolvedTensor): void {
    this.resolved.set(`${this.node.name}::${index}`, tensor);
  }

  /** Resolve a multi-output node's i'th result, registered via `setIndexedOutput`. */
  resolveIndexed(nodeName: string, index: number): ResolvedTensor {
    const tensor = this.resolved.get(`${nodeName}::${index}`);
    if (!tensor) {
      throw new KumaShapeError(
        `Node "${this.node.name}" (getitem) references "${nodeName}"[${index}], which was never registered as a multi-output result.`,
      );
    }
    return tensor;
  }

  private getPipeline(kernelName: string): GPUComputePipeline {
    let pipeline = this.pipelineCache.get(kernelName);
    if (!pipeline) {
      const source = this.kernels.get(kernelName);
      if (!source) {
        throw new KumaManifestError(`.iph package is missing kernel source "kernels/${kernelName}".`);
      }
      const module = this.device.createShaderModule({ code: source });
      pipeline = this.device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
      this.pipelineCache.set(kernelName, pipeline);
    }
    return pipeline;
  }

  /** Record one compute dispatch for `kernelName` against the shared command encoder's
   * compute pass. `buffers` must be in the exact @binding(0..n) order of the kernel's
   * WGSL source. `dispatchElements` is the total number of `out[]` elements the kernel
   * computes — every kernel here uses `@workgroup_size(64)` and folds `gid.y` into its
   * linear index (see engine/dispatch.ts), so dispatches needing more than 65535
   * workgroups spread across a 2D grid instead of failing. */
  dispatchKernel(kernelName: string, buffers: GPUBuffer[], dispatchElements: number): void {
    const pipeline = this.getPipeline(kernelName);
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const totalWorkgroups = Math.max(1, Math.ceil(dispatchElements / KERNEL_WORKGROUP_SIZE));
    const grid = computeDispatchGrid(totalWorkgroups);
    if (!grid) {
      throw new KumaShapeError(
        `Op "${this.node.target}" (node "${this.node.name}") needs ${totalWorkgroups} workgroups, exceeding ` +
          `even a ${MAX_WORKGROUPS_PER_DIMENSION}x${MAX_WORKGROUPS_PER_DIMENSION} 2D dispatch grid's capacity.`,
      );
    }
    this.pass.setPipeline(pipeline);
    this.pass.setBindGroup(0, bindGroup);
    this.pass.dispatchWorkgroups(grid.x, grid.y);
  }
}

export type OpHandler = (ctx: OpContext) => void;
