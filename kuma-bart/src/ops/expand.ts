import type { OpContext } from "../engine/context.js";
import type { ArgValue } from "../types/manifest.js";
import { KumaUnsupportedOpError } from "../errors.js";
import { expandStrides, numElements } from "../engine/shape.js";

const RANK_CAP = 4;

/** aten.expand.default args: (input, size) — broadcast to `size` (entries of -1 mean
 * "keep this axis's existing size"). Implemented via permute.wgsl: broadcast axes get
 * stride 0 (so every output index reads the same single input element along that
 * axis), exactly like permute's own rank-padding trick — no new kernel needed. */
export function expandHandler(ctx: OpContext): void {
  const node = ctx.node;
  const [inputRef, sizeArg] = node.args as [ArgValue, number[]];
  const input = ctx.resolve(inputRef);

  const rankPad = sizeArg.length - input.shape.length;
  const paddedInputShape = new Array(Math.max(0, rankPad)).fill(1).concat(input.shape);
  const targetShape = sizeArg.map((s, d) => (s === -1 ? paddedInputShape[d]! : s));

  if (targetShape.length > RANK_CAP) {
    throw new KumaUnsupportedOpError(
      node.target,
      node.name,
      `expand to rank ${targetShape.length} exceeds the rank-${RANK_CAP} cap shared with permute.wgsl.`,
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
  ctx.setOutput(out, targetShape);
}
