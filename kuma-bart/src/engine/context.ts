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

/** How many copies of each (node, call-within-node) output buffer to keep alive and
 * rotate through, instead of allocating-then-immediately-destroying a fresh one every
 * single call -- this is the fix for real, measured GC pauses (confirmed via Chrome
 * DevTools Performance: "GC surge right before the big lag", correlated with dropped
 * frames on the Frames track) caused by ~270+ fresh GPUBuffer/bind-group/uniform-buffer
 * allocations every frame, at up to 60fps. Must be large enough that, by the time a
 * slot's generation comes back around, the GPU has had a real chance to actually finish
 * with its previous occupant -- see acquireGenerationSlot/releaseGenerationSlot. Not
 * tied to any one model's branch count or shape; this is a fixed rotation depth applied
 * uniformly per (node, call-index), however many of those a given manifest has. */
export const BUFFER_POOL_DEPTH = 3;

/** Persists across every call sharing it (one instance lives on KumaModel, threaded
 * through exactly like pipelineCache/constantCache) -- `pools` is keyed by
 * `${node.name}::${call-index-within-that-node}` (a node can call ctx.createBuffer more
 * than once, e.g. fft.ts's multi-step intermediates), each entry a fixed-size ring of
 * BUFFER_POOL_DEPTH buffers sized once (on first use) to that exact call's byte length
 * -- safe forever after, since Kuma is a static-shape compiler: the same node always
 * produces the same shape, every single call. `callGeneration`/`confirmedGeneration`
 * guard *reuse* safety -- see acquireGenerationSlot. */
export interface BufferPoolState {
  pools: Map<string, GPUBuffer[]>;
  callGeneration: number;
  confirmedGeneration: number;
}

export function createBufferPoolState(): BufferPoolState {
  return { pools: new Map(), callGeneration: 0, confirmedGeneration: -1 };
}

/** Call once, before dispatching anything, from every entry point that shares this
 * pool (runGraph, profileGraph). WebGPU has no per-resource fence -- the only
 * completion signal at all is onSubmittedWorkDone(), measured this session at a real
 * ~30ms latency *independent of how much work or data is involved*. This must never
 * let a call reuse a pooled slot before its previous occupant's GPU work has actually
 * finished, without paying that ~30ms on literally every call (which would undo the
 * whole point of pooling). Usually resolves immediately: confirmedGeneration is kept
 * caught up asynchronously by releaseGenerationSlot below, so by the time a slot's
 * generation comes back around (BUFFER_POOL_DEPTH calls later) it has very likely
 * already been confirmed without this needing to await anything itself. Only actually
 * awaits when the caller is submitting faster than the GPU + its completion-
 * notification latency can keep up with -- the same amortization idea as this
 * session's in-flight pipelining, just now owning a real correctness invariant
 * (no premature buffer reuse) instead of a soft memory-safety heuristic. */
export async function acquireGenerationSlot(device: GPUDevice, pool: BufferPoolState): Promise<void> {
  const reuseGeneration = pool.callGeneration - BUFFER_POOL_DEPTH;
  if (reuseGeneration >= 0 && pool.confirmedGeneration < reuseGeneration) {
    await device.queue.onSubmittedWorkDone();
    pool.confirmedGeneration = pool.callGeneration - 1;
  }
}

/** Call once, right after submit, from the same caller as acquireGenerationSlot.
 * Advances callGeneration for whichever call comes next, and -- not awaited here, on
 * purpose -- kicks off tracking for when *this* call's own GPU work actually
 * completes, so a future acquireGenerationSlot call BUFFER_POOL_DEPTH generations from
 * now finds confirmedGeneration already caught up rather than needing to wait. */
export function releaseGenerationSlot(device: GPUDevice, pool: BufferPoolState): void {
  const myGeneration = pool.callGeneration;
  pool.callGeneration = myGeneration + 1;
  void device.queue.onSubmittedWorkDone().then(() => {
    pool.confirmedGeneration = Math.max(pool.confirmedGeneration, myGeneration);
  });
}

/**
 * Per-node execution context handed to an op handler. Wraps the GPU mechanics
 * (pipeline caching, bind groups, dispatch) so handlers only deal with tensors and
 * shapes. Every kernel binds its buffers in the exact order declared in its WGSL
 * source (storage buffers first, uniform Params last) — `dispatchKernel` relies on
 * callers passing buffers in that same order.
 */
export class OpContext {
  // Counts this node's own uniform()/uniformTyped() calls so each one gets a distinct
  // cache slot below -- some ops (fft.ts's matmulBasis, called repeatedly from
  // complexIfftAlongAxis) build more than one params buffer per node dispatch, all
  // through this same ctx instance, and node name alone would collapse them into one
  // shared (and not necessarily identical) cached buffer.
  private uniformCallIndex = 0;
  // Same rationale as uniformCallIndex, for createBuffer below -- a node can call it
  // more than once (e.g. fft.ts's multi-step intermediates), and each call needs its
  // own pool ring, not a shared one.
  private createBufferCallIndex = 0;

  constructor(
    public readonly device: GPUDevice,
    private readonly kernels: ReadonlyMap<string, string>,
    private readonly pipelineCache: Map<string, GPUComputePipeline>,
    private readonly pass: GPUComputePassEncoder,
    private readonly resolved: Map<string, ResolvedTensor>,
    public readonly node: GraphNode,
    private readonly constantCache: Map<string, GPUBuffer>,
    private readonly bufferPool: BufferPoolState,
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

  /** Returns this call's slot in a BUFFER_POOL_DEPTH-deep ring buffer, keyed by this
   * node + which createBuffer call this is within it -- sized once (lazily, on first
   * use) to `shape`'s byte length, safe to reuse forever after since the same node
   * always produces the same shape on every call (Kuma is a static-shape compiler).
   * Reuse-timing safety across calls is the scheduler's job (acquireGenerationSlot,
   * called once per runGraph/profileGraph call before any createBuffer call can
   * happen) -- by the time this runs, it's already safe to hand back whichever slot
   * `bufferPool.callGeneration % BUFFER_POOL_DEPTH` points at. */
  createBuffer(shape: readonly number[]): GPUBuffer {
    const byteLength = numElements(shape) * 4;
    const key = `${this.node.name}::${this.createBufferCallIndex++}`;
    let slots = this.bufferPool.pools.get(key);
    if (!slots) {
      slots = [];
      for (let i = 0; i < BUFFER_POOL_DEPTH; i++) {
        slots.push(createStorageBuffer(this.device, byteLength));
      }
      this.bufferPool.pools.set(key, slots);
    }
    return slots[this.bufferPool.callGeneration % BUFFER_POOL_DEPTH]!;
  }

  /** A zero-filled buffer of `byteLength` bytes — for bias-less ops (e.g. conv2d/linear with no bias arg). */
  zeros(byteLength: number): GPUBuffer {
    return createStorageBuffer(this.device, byteLength);
  }

  /** Like `zeros`, but cached by `key` — for a zero buffer whose size is itself a fixed
   * (shape-derived) constant reused across calls (e.g. fft.ts's zero bias for its DFT
   * matmuls), so it's allocated once instead of once per inference call. Never written
   * to by any kernel (every binding that uses one is read-only), so sharing one instance
   * across every call site that asks for the same size is safe. */
  getOrCreateZeroBuffer(key: string, byteLength: number): GPUBuffer {
    let buffer = this.constantCache.get(key);
    if (!buffer) {
      buffer = this.zeros(byteLength);
      this.constantCache.set(key, buffer);
    }
    return buffer;
  }

  /** Upload a JS-computed constant (e.g. a DFT basis matrix — see engine/dft.ts) as a
   * storage buffer, ready to bind as a kernel input like any other tensor. */
  uploadConstant(data: Float32Array): GPUBuffer {
    return uploadFloat32(this.device, data);
  }

  /** Like `uploadConstant`, but only computes + uploads once per `key`, ever — reused
   * across every node and every subsequent inference call on this model. For data
   * that's a pure function of a fixed shape, never of the actual input (e.g. a DFT
   * basis matrix keyed by transform length): recomputing and re-uploading it on every
   * single frame is pure waste, not extra correctness. `key` is caller-namespaced (e.g.
   * "complexIfft:cos:180") since this cache is shared across every op, not per-kernel. */
  getOrUploadConstant(key: string, compute: () => Float32Array): GPUBuffer {
    let buffer = this.constantCache.get(key);
    if (!buffer) {
      buffer = this.uploadConstant(compute());
      this.constantCache.set(key, buffer);
    }
    return buffer;
  }

  /** Pack u32 fields (WGSL struct order) into a ready-to-bind uniform buffer. Cached
   * forever once built: every field here comes from graph structure (shape, stride,
   * weight dims, etc. -- see node.meta/args), never from actual input data, so the
   * exact same buffer is correct on every subsequent call for this node -- rebuilding
   * it every frame is pure waste, not extra correctness. Safe to cache indefinitely:
   * never written to again after creation, so there's no multi-frame-in-flight reuse
   * hazard the way pooled *output* buffers have. */
  uniform(fields: readonly number[]): GPUBuffer {
    const key = `params:${this.node.name}:${this.uniformCallIndex++}`;
    let buffer = this.constantCache.get(key);
    if (!buffer) {
      buffer = createUniformBuffer(this.device, packParams(fields));
      this.constantCache.set(key, buffer);
    }
    return buffer;
  }

  /** Like `uniform`, for Params structs that mix u32 and f32 fields (e.g. clamp,
   * group_norm/layer_norm's eps). Same caching rationale as `uniform`. */
  uniformTyped(fields: readonly TypedParamField[]): GPUBuffer {
    const key = `params:${this.node.name}:${this.uniformCallIndex++}`;
    let buffer = this.constantCache.get(key);
    if (!buffer) {
      buffer = createUniformBuffer(this.device, packTypedParams(fields));
      this.constantCache.set(key, buffer);
    }
    return buffer;
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

  private bindAndDispatch(kernelName: string, buffers: GPUBuffer[], gridX: number, gridY: number, gridZ: number): void {
    const pipeline = this.getPipeline(kernelName);
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    this.pass.setPipeline(pipeline);
    this.pass.setBindGroup(0, bindGroup);
    this.pass.dispatchWorkgroups(gridX, gridY, gridZ);
  }

  /** Record one compute dispatch for `kernelName` against the shared command encoder's
   * compute pass. `buffers` must be in the exact @binding(0..n) order of the kernel's
   * WGSL source. `dispatchElements` is the total number of `out[]` elements the kernel
   * computes — every kernel using this entry point uses `@workgroup_size(64)` and folds
   * `gid.y` into its linear index (see engine/dispatch.ts), so dispatches needing more
   * than 65535 workgroups spread across a 2D grid instead of failing. Kernels with a
   * genuinely 2D notion of work (e.g. a tiled matmul, where workgroup_id.(x,y) means
   * "which output tile" rather than folding into one linear index) should use
   * `dispatchKernel2D` instead — its grid shape carries real meaning to the shader. */
  dispatchKernel(kernelName: string, buffers: GPUBuffer[], dispatchElements: number): void {
    const totalWorkgroups = Math.max(1, Math.ceil(dispatchElements / KERNEL_WORKGROUP_SIZE));
    const grid = computeDispatchGrid(totalWorkgroups);
    if (!grid) {
      throw new KumaShapeError(
        `Op "${this.node.target}" (node "${this.node.name}") needs ${totalWorkgroups} workgroups, exceeding ` +
          `even a ${MAX_WORKGROUPS_PER_DIMENSION}x${MAX_WORKGROUPS_PER_DIMENSION} 2D dispatch grid's capacity.`,
      );
    }
    this.bindAndDispatch(kernelName, buffers, grid.x, grid.y, 1);
  }

  /** Like `dispatchKernel`, but for a kernel whose `@builtin(workgroup_id)` is read
   * directly as grid coordinates (2D or 3D) rather than folded into a linear element
   * index -- `gridX`/`gridY`/`gridZ` are workgroup *counts* along each axis (e.g.
   * ceil(N/TILE), ceil(M/TILE) for a tiled matmul; or ceil(W/TILE), ceil(H/TILE),
   * batch*channels for a tiled conv), not element counts. `gridZ` defaults to 1 for
   * kernels that only need a 2D grid. */
  dispatchKernelGrid(kernelName: string, buffers: GPUBuffer[], gridX: number, gridY: number, gridZ: number = 1): void {
    if (gridX > MAX_WORKGROUPS_PER_DIMENSION || gridY > MAX_WORKGROUPS_PER_DIMENSION || gridZ > MAX_WORKGROUPS_PER_DIMENSION) {
      throw new KumaShapeError(
        `Op "${this.node.target}" (node "${this.node.name}") needs a ${gridX}x${gridY}x${gridZ} workgroup grid, ` +
          `exceeding the ${MAX_WORKGROUPS_PER_DIMENSION}-per-dimension dispatch limit.`,
      );
    }
    this.bindAndDispatch(kernelName, buffers, Math.max(1, gridX), Math.max(1, gridY), Math.max(1, gridZ));
  }
}

export type OpHandler = (ctx: OpContext) => void;
