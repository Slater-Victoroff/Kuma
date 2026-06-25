import type { OpContext, ResolvedTensor } from "../engine/context.js";
import type { ArgValue } from "../types/manifest.js";
import { KumaUnsupportedOpError } from "../errors.js";
import { expandStrides, numElements } from "../engine/shape.js";

const RANK_CAP = 4;

/** Materializes `input` broadcast up to `targetShape` (right-aligned, NumPy/PyTorch
 * rules) via permute.wgsl: broadcast axes get stride 0, so every output index along
 * that axis reads the same single input element — exactly aten.expand.default's own
 * trick, factored out here because binaryElementwise (ops/elementwise.ts) needs the
 * same materialization for any add/mul/sub/div/minimum whose operands implicitly
 * broadcast. PyTorch traces implicit broadcasting as a plain aten.*.Tensor node with no
 * expand in between (ATen's own binary kernels broadcast internally) — but this
 * project's binary WGSL kernels are flat/broadcast-unaware, so kuma-bart has to
 * materialize the broadcast itself before they ever see the buffers. */
export function broadcastTensor(ctx: OpContext, input: ResolvedTensor, targetShape: readonly number[]): GPUBuffer {
  if (targetShape.length > RANK_CAP) {
    throw new KumaUnsupportedOpError(
      ctx.node.target,
      ctx.node.name,
      `broadcasting to rank ${targetShape.length} exceeds the rank-${RANK_CAP} cap shared with permute.wgsl.`,
    );
  }

  const strides = expandStrides(input.shape, targetShape);
  const pad = RANK_CAP - targetShape.length;
  const paddedOutShape = new Array(pad).fill(1).concat(targetShape);
  const paddedStrides = new Array(pad).fill(0).concat(strides);
  const n = numElements(targetShape);

  const out = ctx.createBuffer(targetShape);
  const params = ctx.uniform([...paddedOutShape, ...paddedStrides, n]);
  ctx.dispatchKernel("permute.wgsl", [input.buffer, out, params], n);
  return out;
}

/** aten.expand.default args: (input, size) — broadcast to `size` (entries of -1 mean
 * "keep this axis's existing size"). */
export function expandHandler(ctx: OpContext): void {
  const node = ctx.node;
  const [inputRef, sizeArg] = node.args as [ArgValue, number[]];
  const input = ctx.resolve(inputRef);

  const rankPad = sizeArg.length - input.shape.length;
  const paddedInputShape = new Array(Math.max(0, rankPad)).fill(1).concat(input.shape);
  const targetShape = sizeArg.map((s, d) => (s === -1 ? paddedInputShape[d]! : s));

  const out = broadcastTensor(ctx, input, targetShape);
  // Same complex-pairing gap as ops/gather.ts/gather_dim.ts -- broadcasting is just a
  // stride-0 buffer gather, so the imaginary half (if any) needs the identical
  // broadcast, or it silently vanishes.
  const outImag = input.imag
    ? broadcastTensor(ctx, { buffer: input.imag, shape: input.shape }, targetShape)
    : undefined;
  ctx.setOutput(out, targetShape, outImag);
}
