import type { OpContext } from "../engine/context.js";
import type { ArgValue } from "../types/manifest.js";
import { clampSliceBounds, normalizeDim, numElements, outerInner } from "../engine/shape.js";

/** aten.slice.Tensor args: (input, dim, start, end, step). PyTorch FX commonly encodes
 * "to the end" as a huge sentinel int for `end` and `null` for default start/end. */
export function sliceHandler(ctx: OpContext): void {
  const node = ctx.node;
  const args = node.args;
  const inputRef = args[0] as ArgValue;
  const dimArg = args[1] as number;
  const startArg = args[2] as number | null;
  const endArg = args[3] as number | null;
  const stepArg = args[4] as number | null | undefined;
  const input = ctx.resolve(inputRef);
  const dim = normalizeDim(dimArg, input.shape.length);
  const dimSize = input.shape[dim]!;
  const { start, end } = clampSliceBounds(startArg, endArg, dimSize);
  const step = stepArg ?? 1;
  const outExtent = step > 0 ? Math.max(0, Math.ceil((end - start) / step)) : 0;

  const outShape = input.shape.slice();
  outShape[dim] = outExtent;
  const { outer, inner } = outerInner(input.shape, dim);
  const n = numElements(outShape);

  const out = ctx.createBuffer(outShape);
  const params = ctx.uniform([outer, inner, dimSize, outExtent, start, step, n]);
  ctx.dispatchKernel("slice.wgsl", [input.buffer, out, params], n);
  ctx.setOutput(out, outShape);
}
