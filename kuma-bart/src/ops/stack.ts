import type { OpContext, ResolvedTensor } from "../engine/context.js";
import type { ArgValue } from "../types/manifest.js";
import { KumaShapeError } from "../errors.js";
import { normalizeDim } from "../engine/shape.js";
import { dispatchConcatChain } from "./concat.js";

function unsqueezeAt(t: ResolvedTensor, dim: number): ResolvedTensor {
  const shape = t.shape.slice();
  shape.splice(dim, 0, 1);
  return { buffer: t.buffer, shape, imag: t.imag };
}

/** aten.stack.default args: ([tensors], dim) — equivalent to
 * cat([t.unsqueeze(dim) for t in tensors], dim); the unsqueeze is metadata-only (same
 * buffer, shape with a 1 inserted), so this reuses concat's dispatch chain unchanged —
 * no new kernel needed. */
export function stackHandler(ctx: OpContext): void {
  const node = ctx.node;
  const [listArg, dimArg] = node.args as [ArgValue[], number];
  if (!Array.isArray(listArg) || listArg.length < 1) {
    throw new KumaShapeError(
      `Op "${node.target}" (node "${node.name}") needs at least 1 tensor to stack, got ` +
        `${Array.isArray(listArg) ? listArg.length : "a non-list args[0]"}.`,
    );
  }

  const tensors = listArg.map((ref) => ctx.resolve(ref));
  // stack inserts a new axis, so the valid dim range is rank+1, not rank.
  const dim = normalizeDim(dimArg, tensors[0]!.shape.length + 1);
  const unsqueezed = tensors.map((t) => unsqueezeAt(t, dim));
  const result = dispatchConcatChain(ctx, unsqueezed, dim);

  const outShape = node.meta.shape ?? result.shape;
  ctx.setOutput(result.buffer, outShape);
}
